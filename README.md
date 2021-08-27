## Compile protos
```bash
# make protos/common_pb2.py
python -m grpc_tools.protoc \
    -I . \
    --python_out . \
    protos/common.proto

# make protos/face_detect_pb2.py and protos/face_detect_pb2_grpc.py
python -m grpc_tools.protoc \
    -I . \
    --python_out . \
    --grpc_python_out . \
    protos/face_detect.proto
```

## Run server
```bash
python3 face_detect_seer.py
```

## Run client demo
```bash
python3 face_detect_client.py path/to/image/file
```
