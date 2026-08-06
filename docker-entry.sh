#!/bin/sh
# pi-vision-struct 容器入口：`docker run image <action> [args]`
# action: pix / ocr / layout / omniparser / pdf / rules / audit / critic /
#         cluster / analyze / capture / pptx / dom / semantic / wallpaper / env
# 或直接传脚本路径: `docker run image python/vs_pix.py --image /work/x.png`
TOOLS=/opt/pi-vision-struct/python

if [ "$1" = "help" ] || [ -z "$1" ]; then
  echo "pi-vision-struct container"
  echo "用法: docker run --rm -v \$PWD:/work pi-vision-struct:latest <action> [args]"
  echo "actions: pix ocr layout omniparser pdf rules audit critic cluster analyze"
  echo "         capture pptx dom semantic wallpaper env"
  exit 0
fi

# 直接脚本路径
if [ -f "$TOOLS/$1" ]; then
  SCRIPT="$1"; shift
  exec python -u "$TOOLS/$SCRIPT" "$@"
fi

ACTION="$1"; shift
case "$ACTION" in
  env) exec python -u -c "import PIL,numpy,onnxruntime,rapidocr,pptx,fitz,mss,torch,transformers,ultralytics,paddle,paddleocr,paddlex,open_clip; import sys; print('python', sys.version.split()[0], '| all deps OK')" ;;
  capture) exec python -u "$TOOLS/vs_capture.py" "$@" ;;
  *) exec python -u "$TOOLS/vs_${ACTION}.py" "$@" ;;
esac
