import json
import logging
import os
import pathlib
import shutil
import tempfile
from enum import Enum
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from requests_toolbelt import MultipartEncoder
from starlette.background import BackgroundTasks

from monailabel.interfaces import MONAILabelApp
from monailabel.utils.others.generic import get_app_instance, get_mime_type

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/inference",
    tags=["AppService"],
    responses={
        404: {"description": "Not found"},
        200: {
            "description": "OK",
            "content": {
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "points": {
                                "type": "string",
                                "description": "Reserved for future; Currently it will be empty",
                            },
                            "file": {
                                "type": "string",
                                "format": "binary",
                                "description": "The result NIFTI image which will have segmentation mask",
                            },
                        },
                    },
                    "encoding": {
                        "points": {"contentType": "text/plain"},
                        "file": {"contentType": "application/octet-stream"},
                    },
                },
                "application/json": {"schema": {"type": "string", "example": "{}"}},
                "application/octet-stream": {"schema": {"type": "string", "format": "binary"}},
            },
        },
    },
)


class ResultType(str, Enum):
    image = "image"
    json = "json"
    all = "all"


def remove_file(path: str) -> None:
    if os.path.exists(path):
        os.unlink(path)


def send_response(result, output, background_tasks):
    res_img = result.get("label")
    res_json = result.get("params")

    if output == "json":
        return res_json

    background_tasks.add_task(remove_file, res_img)
    m_type = get_mime_type(res_img)

    if output == "image":
        return FileResponse(res_img, media_type=m_type, filename=os.path.basename(res_img))

    res_fields = dict()
    res_fields["params"] = (None, json.dumps(res_json), "application/json")
    res_fields["image"] = (os.path.basename(res_img), open(res_img, "rb"), m_type)

    return_message = MultipartEncoder(fields=res_fields)
    return Response(content=return_message.to_string(), media_type=return_message.content_type)


# TODO:: Define request uri for (model, image, params)
@router.post("/{model}", summary="Run Inference for supported model")
async def run_inference(
    background_tasks: BackgroundTasks,
    model: str,
    image: str,
    params: str = Form("{}"),
    label: UploadFile = File(None),
    output: Optional[ResultType] = None,
):
    request = {"model": model, "image": image}

    if label:
        file_ext = "".join(pathlib.Path(label.filename).suffixes) if label.filename else ".nii.gz"
        label_file = tempfile.NamedTemporaryFile(suffix=file_ext).name

        with open(label_file, "wb") as buffer:
            shutil.copyfileobj(label.file, buffer)
            request["label"] = label_file
            background_tasks.add_task(remove_file, label_file)

    instance: MONAILabelApp = get_app_instance()
    config = instance.info().get("config", {}).get("infer", {})
    request.update(config)

    p = json.loads(params) if params else {}
    request.update(p)

    logger.info(f"Infer Request: {request}")
    result = instance.infer(request)
    if result is None:
        raise HTTPException(status_code=500, detail="Failed to execute infer")
    return send_response(result, output, background_tasks)
