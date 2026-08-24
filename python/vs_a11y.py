#!/usr/bin/env python3
"""vs_a11y.py — L0 传感器：Linux AT-SPI 无障碍树 → schema v2 元素（只读）。

无障碍树是桌面原生应用自报的结构真值（角色/名称/状态/屏幕坐标），是原生应用的
「DOM 等价物」；比截图+OCR 更准，与 dom/pptx 同属 L0 源码层。

用法:
  vs_a11y.py --list                        # 列出当前可见应用
  vs_a11y.py [--app NAME] [--max-elements 80] [--with-text]

依赖: 宿主 python3 + at-spi2-core + python-gobject。
      （conda env 无 gi 模块，本脚本固定用系统解释器运行）
      Arch 安装: sudo pacman -S at-spi2-core python-gobject
坐标系: screen_px（全屏绝对坐标）。只读保证: 仅调用 get_* 只读接口。
"""
import argparse
import json
import sys
from typing import Any

import vs_schema as S

MAX_DEPTH = 14
TEXT_ROLES = {"document text", "text", "entry", "terminal", "paragraph"}

# 延迟绑定（main 内成功导入后赋值）：Atspi 模块与坐标类型
Atspi = None  # type: ignore[assignment]
SCREEN = None  # Atspi.CoordType.SCREEN


def fail(msg: str, hint: str = "") -> int:
    print(json.dumps({"error": "vs_a11y failed", "detail": msg, "hint": hint},
                     ensure_ascii=False))
    return 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="仅列出可见应用")
    ap.add_argument("--app", help="只导出指定应用名（默认全部）")
    ap.add_argument("--max-elements", type=int, default=80)
    ap.add_argument("--with-text", action="store_true",
                    help="对文本类角色额外抓取内容（前 300 字符）")
    args = ap.parse_args()

    # gi 只在宿主系统解释器存在；延迟导入以输出可操作的安装指引
    try:
        import gi  # type: ignore[import-not-found]
        gi.require_version("Atspi", "2.0")
        global Atspi, SCREEN
        from gi.repository import Atspi as _Atspi  # type: ignore[import-not-found]
        Atspi = _Atspi
        SCREEN = Atspi.CoordType.SCREEN
    except (ImportError, ValueError) as e:
        return fail(f"AT-SPI 绑定不可用: {e}",
                    "安装: sudo pacman -S at-spi2-core python-gobject "
                    "(Debian: libatspi2.0-0 python3-gi)")

    try:
        desktop = Atspi.get_desktop(0)

        def app_name(i: int) -> str:
            try:
                return desktop.get_child_at_index(i).get_name() or f"app{i}"
            except Exception:
                return f"app{i}"

        if args.list:
            apps = [app_name(i) for i in range(desktop.get_child_count())]
            print(S.dump_json({"schema": S.SCHEMA, "task": "a11y-list",
                               "apps": apps}))
            return 0

        elements: list[dict[str, Any]] = []
        truncated = False
        next_id = 0

        def walk(acc: Any, depth: int) -> None:
            nonlocal next_id, truncated
            if truncated or depth > MAX_DEPTH or next_id >= args.max_elements:
                truncated = truncated or next_id >= args.max_elements
                return
            try:
                role = acc.get_role_name()
                name = acc.get_name() or ""
                ext = acc.get_extents(SCREEN)
            except Exception:
                return
            x2, y2 = ext.x + max(ext.width, 0), ext.y + max(ext.height, 0)
            visible = bool(ext.width) and bool(ext.height)
            el = S.element(next_id, role, [ext.x, ext.y, x2, y2],
                           text=name[:120] or None,
                           coordsys="screen_px")
            meta: dict[str, Any] = {}
            if args.with_text and role.lower() in TEXT_ROLES:
                try:
                    txt = acc.get_text_interface()
                    if txt is not None:
                        meta["content"] = txt.get_text(0, 300) or ""
                except Exception:
                    pass
            if meta:
                el["meta"] = meta
            elements.append(el)
            next_id += 1
            if not visible:
                return
            try:
                n = acc.get_child_count()
            except Exception:
                return
            for i in range(n):
                try:
                    child = acc.get_child_at_index(i)
                except Exception:
                    continue
                if child is not None:
                    walk(child, depth + 1)

        apps_range = range(desktop.get_child_count())
        for i in apps_range:
            name = app_name(i)
            if args.app and args.app.lower() not in name.lower():
                continue
            walk(desktop.get_child_at_index(i), 0)

        report = S.envelope(task="a11y", sensors=["atspi"], coordsys="screen_px",
                            source={"type": "a11y", "app": args.app or "*"})
        report["elements"] = elements
        report["truncated"] = truncated
        report["notation"] = S.NOTATION_GUIDE
        report["metrics"] = {"element_count": len(elements)}
        print(S.dump_json(report))
        return 0
    except Exception as e:
        return fail(str(e)[:500])


if __name__ == "__main__":
    sys.exit(main())
