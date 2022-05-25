#!/bin/bash

set -e
docker build -t qhduan/onnx-wenzhong-gen:0.2 .
docker push qhduan/onnx-wenzhong-gen:0.2

