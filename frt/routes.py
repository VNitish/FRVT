from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from starlette.status import HTTP_200_OK, HTTP_500_INTERNAL_SERVER_ERROR
from fastapi import APIRouter, Depends

import inspect
from Utilities.config import *
from Utilities.authentication import *
import traceback
from frt import service
from frt.faiss import *
import os

MODULE_NAME = "routes"
FILE_NAME = os.path.basename(__file__)


router = APIRouter()

@router.on_event("startup")
async def on_startup():
    initialize_faiss_indexes()

@router.get("/ping")
async def ping():
    return {"message": "pong"}

@router.post("/login")
def login(input: dict):
    email = input.get("email")
    password = input.get("password")

    user = db.users.find_one({"email": email})
    if not user or not verify_password(password, user["password"]):
        return JSONResponse(
            status_code=401,
            content={
                "data": {
                    "access_token": ""
                },
                "message": "Invalid Credentials",
                "status": False
            }
        )
    
    expires_delta = Config.ACCESS_TOKEN_EXPIRE_MINUTES
    token, expire = create_access_token(
        data={"sub": user["email"]},
        expires_delta=expires_delta
    )

    return JSONResponse(
        content={
            "data": {
                "access_token": token,
                "expires_at": expire.strftime("%a, %d %b %Y %H:%M:%S GMT")
            },
            "message": "Logged-In Successfully",
            "status": True
        }
    )

@router.post("/verify")
async def verify_route(request: Request, user=Depends(get_current_user)):
    method_name = inspect.stack()[0][3]
    method, path, ip = request.method, request.url.path, request.client.host

    try:
        input_args = await request.json()
        response = await service.verify_response(input_args)
    except Exception as e:
        logger.error(f"Exception Message: {traceback.format_exc()}\n, File-Name: {FILE_NAME}, "
                     f"Method-Name: {method_name}")
        logger.info(20 * "````")
        response = {
            "status": False,
            "data": {},
            "message": EXCEPTION_MESSAGE,
            "errorMessage": str(e),
            "code": EXCEPTION_CODE,
            "is_exception": True
        }

    return_payload = return_process(
        data=input_args,
        response=response,
        method=method,
        path=path,
        ip=ip,
        method_name=method_name,
        file_name=FILE_NAME,
        module_name=MODULE_NAME.capitalize(),
        code=response.get("code"),
    )
    status_code = HTTP_200_OK if response.get("status") else HTTP_500_INTERNAL_SERVER_ERROR
    return JSONResponse(content=return_payload, status_code=status_code)

@router.post("/create-collection")
async def create_collection(request: Request, user=Depends(get_current_user)):
    method_name = inspect.stack()[0][3]
    method, path, ip = request.method, request.url.path, request.client.host

    try:
        input_args = await request.json()
        response = await service.create_collection_response(input_args)
    except Exception as e:
        logger.error(
            f"Exception Message: {traceback.format_exc()}\n, File-Name: {FILE_NAME}, "
            f"Method-Name: {method_name}"
        )
        logger.info(20 * "````")
        response = {
            "status": False,
            "data": {},
            "message": EXCEPTION_MESSAGE,
            "errorMessage": str(e),
            "code": EXCEPTION_CODE,
            "is_exception": True
        }

    return_response = return_process(
        data=input_args,
        response=response,
        method=method,
        path=path,
        ip=ip,
        method_name=method_name,
        file_name=FILE_NAME,
        module_name=MODULE_NAME.capitalize(),
        code=response["code"]
    )

    status_code = HTTP_200_OK if response["status"] else HTTP_500_INTERNAL_SERVER_ERROR
    return JSONResponse(content=return_response, status_code=status_code)

@router.get("/list-collections")
async def list_collections(request: Request, user=Depends(get_current_user)):
    method_name = inspect.stack()[0][3]
    method, path, ip = request.method, request.url.path, request.client.host
    input_args = {}

    try:
        response = await service.list_collections_response()
    except Exception as e:
        logger.error(
            f"Exception Message: {traceback.format_exc()}\n, File-Name: {FILE_NAME}, Method-Name: {method_name}"
        )
        logger.info(20 * "````")
        response = {
            "status": False,
            "data": {},
            "message": EXCEPTION_MESSAGE,
            "errorMessage": str(e),
            "code": EXCEPTION_CODE,
            "is_exception": True,
        }

    return_payload = return_process(
        data=input_args,
        response=response,
        method=method,
        path=path,
        ip=ip,
        method_name=method_name,
        file_name=FILE_NAME,
        module_name=MODULE_NAME.capitalize(),
        code=response["code"],
    )

    return JSONResponse(
        content=return_payload,
        status_code=HTTP_200_OK if response["status"] else HTTP_500_INTERNAL_SERVER_ERROR,
    )

@router.delete("/delete-collection")
async def delete_collection(request: Request, user=Depends(get_current_user)):
    method_name = inspect.stack()[0][3]
    method, path, ip = request.method, request.url.path, request.client.host

    try:
        input_args = await request.json()
        response = await service.delete_collection_response(input_args)
    except Exception as e:
        logger.error(
            f"Exception Message: {traceback.format_exc()}\n, File-Name: {FILE_NAME}, Method-Name: {method_name}"
        )
        logger.info(20 * "````")
        response = {
            "status": False,
            "data": {},
            "message": EXCEPTION_MESSAGE,
            "errorMessage": str(e),
            "code": EXCEPTION_CODE,
            "is_exception": True,
        }

    return_payload = return_process(
        data=input_args,
        response=response,
        method=method,
        path=path,
        ip=ip,
        method_name=method_name,
        file_name=FILE_NAME,
        module_name=MODULE_NAME.capitalize(),
        code=response["code"],
    )

    return JSONResponse(
        content=return_payload,
        status_code=HTTP_200_OK if response["status"] else HTTP_500_INTERNAL_SERVER_ERROR,
    )

@router.get("/list-images")
async def list_images(request: Request, collection_name: str, user=Depends(get_current_user)):
    method_name = inspect.stack()[0][3]
    method, path, ip = request.method, request.url.path, request.client.host
    input_args = {"collection_name": collection_name}

    try:
        response = await service.list_images_response(collection_name)
    except Exception as e:
        logger.error(
            f"Exception Message: {traceback.format_exc()}\n, File-Name: {FILE_NAME}, Method-Name: {method_name}"
        )
        logger.info(20 * "````")
        response = {
            "status": False,
            "data": {},
            "message": EXCEPTION_MESSAGE,
            "errorMessage": str(e),
            "code": EXCEPTION_CODE,
            "is_exception": True,
        }

    return_payload = return_process(
        data=input_args,
        response=response,
        method=method,
        path=path,
        ip=ip,
        method_name=method_name,
        file_name=FILE_NAME,
        module_name=MODULE_NAME.capitalize(),
        code=response["code"],
    )

    return JSONResponse(
        content=return_payload,
        status_code=HTTP_200_OK if response["status"] else HTTP_500_INTERNAL_SERVER_ERROR,
    )

@router.delete("/delete-image")
async def delete_image(request: Request, user=Depends(get_current_user)):
    method_name = inspect.stack()[0][3]
    method, path, ip = request.method, request.url.path, request.client.host

    try:
        input_args = await request.json()
        response = await service.delete_image_response(input_args)
    except Exception as e:
        logger.error(
            f"Exception Message: {traceback.format_exc()}\n, File-Name: {FILE_NAME}, Method-Name: {method_name}"
        )
        logger.info(20 * "````")
        response = {
            "status": False,
            "data": {},
            "message": EXCEPTION_MESSAGE,
            "errorMessage": str(e),
            "code": EXCEPTION_CODE,
            "is_exception": True,
        }

    return_payload = return_process(
        data=input_args,
        response=response,
        method=method,
        path=path,
        ip=ip,
        method_name=method_name,
        file_name=FILE_NAME,
        module_name=MODULE_NAME.capitalize(),
        code=response["code"],
    )

    return JSONResponse(
        content=return_payload,
        status_code=HTTP_200_OK if response["status"] else HTTP_500_INTERNAL_SERVER_ERROR,
    )


@router.post("/add-image")
async def add_image(request: Request, user=Depends(get_current_user)):
    method_name = inspect.stack()[0][3]
    method, path, ip = request.method, request.url.path, request.client.host

    try:
        input_args = await request.json()
        response = await service.add_image_response(input_args)
    except Exception as e:
        logger.error(f"Exception Message: {traceback.format_exc()}\n, File-Name: {FILE_NAME}, "
                     f"Method-Name: {method_name}")
        logger.info(20 * "````")
        response = {
            "status": False,
            "data": {},
            "message": EXCEPTION_MESSAGE,
            "errorMessage": str(e),
            "code": EXCEPTION_CODE,
            "is_exception": True
        }

    return_response = return_process(
        data=input_args,
        response=response,
        method=method,
        path=path,
        ip=ip,
        method_name=method_name,
        file_name=FILE_NAME,
        module_name=MODULE_NAME.capitalize(),
        code=response['code']
    )
    if response.get("status") is False:
        status_code = 500
    else:
        if response.get("code") == BAD_REQUEST_CODE:
            status_code = 400
        elif response.get("code") == CREATED_CODE:
            status_code = 201
        elif response.get("code") == ALREADY_EXISTS_CODE:
            status_code = 409
        else:
            status_code = 200  # Default

    return JSONResponse(
        content=return_response,
        status_code=status_code
    )

@router.post("/find")
async def find_route(request: Request, user=Depends(get_current_user)):
    input_args = await request.json()
    method_name = inspect.stack()[0][3]
    method, path, ip = request.method, request.url.path, request.client.host
    
    try:
        response = await service.find_response(input_args)
        logging.info(f"find response:{response}")
    except Exception as e:
        logging.error(
            f"Exception Message: {traceback.format_exc()}\n"
            f"File-Name: {FILE_NAME}, Method-Name: {method_name}"
        )
        response = {
            "status": False,
            "data": {},
            "message": EXCEPTION_MESSAGE,
            "errorMessage": str(e),
            "code": EXCEPTION_CODE,
            "is_exception": True,
        }

    return_payload = return_process(
        data=input_args,
        response=response,
        method=method,
        path=path,
        ip=ip,
        method_name=method_name,
        file_name=FILE_NAME,
        module_name=MODULE_NAME.capitalize(),
        code=response.get("code"),
    )
    if response.get("status") is False:
        status_code = 500
    else:
        # If BAD_REQUEST_CODE or other custom mappings, return accordingly
        if response.get("code") == BAD_REQUEST_CODE:
            status_code = 400
        elif response.get("code") == CREATED_CODE:
            status_code = 201  # Or 201 if appropriate
        else:
            status_code = 200  # Default

    return JSONResponse(content=return_payload, status_code=status_code)

@router.post("/rebuild-index")
async def rebuild_index_route(request: Request, user=Depends(get_current_user)):
    input_args = await request.json()
    method_name = inspect.stack()[0][3]
    method, path, ip = request.method, request.url.path, request.client.host
    
    try:
        response = await service.rebuid_index_response(input_args)
        logging.info(f"find response:{response}")
    except Exception as e:
        logging.error(
            f"Exception Message: {traceback.format_exc()}\n"
            f"File-Name: {FILE_NAME}, Method-Name: {method_name}"
        )
        response = {
            "status": False,
            "data": {},
            "message": EXCEPTION_MESSAGE,
            "errorMessage": str(e),
            "code": EXCEPTION_CODE,
            "is_exception": True,
        }

    return_payload = return_process(
        data=input_args,
        response=response,
        method=method,
        path=path,
        ip=ip,
        method_name=method_name,
        file_name=FILE_NAME,
        module_name=MODULE_NAME.capitalize(),
        code=response.get("code"),
    )
    if response.get("status") is False:
        status_code = 500
    else:
        # If BAD_REQUEST_CODE or other custom mappings, return accordingly
        if response.get("code") == BAD_REQUEST_CODE:
            status_code = 400
        elif response.get("code") == CREATED_CODE:
            status_code = 201  # Or 201 if appropriate
        else:
            status_code = 200  # Default

    return JSONResponse(content=return_payload, status_code=status_code)

@router.post("/detect-face")
async def detect_face_route(request: Request, user=Depends(get_current_user)):
    input_args = await request.json()
    method_name = inspect.stack()[0][3]
    method, path, ip = request.method, request.url.path, request.client.host
    
    try:
        response = await service.detect_face_response(input_args)
    except Exception as e:
        logging.error(
            f"Exception Message: {traceback.format_exc()}\n"
            f"File-Name: {FILE_NAME}, Method-Name: {method_name}"
        )
        response = {
            "status": False,
            "data": {},
            "message": EXCEPTION_MESSAGE,
            "errorMessage": str(e),
            "code": EXCEPTION_CODE,
            "is_exception": True,
        }

    return_payload = return_process(
        data=input_args,
        response=response,
        method=method,
        path=path,
        ip=ip,
        method_name=method_name,
        file_name=FILE_NAME,
        module_name=MODULE_NAME.capitalize(),
        code=response.get("code"),
    )
    status_code = 200 if response.get("status") else 500
    return JSONResponse(content=return_payload, status_code=status_code)
