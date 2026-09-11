#!/usr/bin/env python3
"""Generate the Starlight API reference from pyfor's docstrings.

Reads the package statically with griffe (no imports, so the heavy runtime
dependencies are not needed) and writes one Markdown page per module into
``docs/src/content/docs/api/``. Page anchors keep the dotted Sphinx form,
for example ``#pyfor.rasterizer.GridSpec``, so links from the old site and from
the hand written pages keep working.

Usage: python docs/scripts/generate_api.py
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

import griffe
from griffe import Parser

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE = "pyfor"
MODULES = [
    "clip",
    "cloud",
    "collection",
    "gisexport",
    "ground_filter",
    "metrics",
    "rasterizer",
    "voxelizer",
]
OUT_DIR = REPO_ROOT / "docs" / "src" / "content" / "docs" / "api"

 
# be handled rather than silently dropped.
RENDERED_SECTIONS = {
    "text",
    "parameters",
    "returns",
    "yields",
    "receives",
    "raises",
    "warns",
    "attributes",
    "examples",
    "admonition",
    "deprecated",
}
SKIPPED_MEMBERS: set[str] = set()
UNRENDERED_SECTIONS: set[str] = set()


RST_ROLE_RE = re.compile(
    r":(?:py:)?(?:class|func|meth|attr|mod|data|obj|ref|doc|exc|const|term|numref):"
    r"`~?\.?([^`]+)`"
)
RST_LINK_RE = re.compile(r"`([^`<]+?)\s*<([^>]+)>`_{1,2}")
RST_FIELD_RE = re.compile(r":(?:param|parameter|keyword|type)\s+([^:]+):|:returns?:|:rtype:")


def clean_rst(text: str) -> str:
    """Turn the Sphinx inline roles used in the docstrings into Markdown."""
    text = RST_LINK_RE.sub(r"[\1](\2)", text)
    text = RST_ROLE_RE.sub(r"`\1`", text)
    return text


def escape_prose(text: str) -> str:
    """Collapse whitespace and escape angle brackets, leaving code spans alone."""
    text = clean_rst(text)
    out: list[str] = []
    pending_space = False
    i = 0
    while i < len(text):
        char = text[i]
        if char == "`":
            run = i
            while run < len(text) and text[run] == "`":
                run += 1
            fence = text[i:run]
            end = text.find(fence, run)
            if end == -1:
                out.append(fence)
                i = run
                continue
            if pending_space and out:
                out.append(" ")
            out.append(text[i : end + len(fence)])
            pending_space = False
            i = end + len(fence)
        elif char.isspace():
            pending_space = True
            i += 1
        else:
            if pending_space and out:
                out.append(" ")
            pending_space = False
            out.append({"<": "&lt;", ">": "&gt;"}.get(char, char))
            i += 1
    return "".join(out).strip()


def cell(text: str) -> str:
    """Escape a value for a Markdown table cell."""
    return escape_prose(text).replace("|", "\\|")


def paragraph(text: str) -> str:
    """Collapse a docstring paragraph to the single line Markdown expects."""
    return escape_prose(text)


def render_text(text: str) -> list[str]:
    """Render a docstring text section, keeping its paragraph breaks."""
    blocks = [block for block in re.split(r"\n\s*\n", text) if block.strip()]
    lines: list[str] = []
    for block in blocks:
        lines += [paragraph(block), ""]
    return lines


def annotation_of(param_or_return) -> str:
    annotation = getattr(param_or_return, "annotation", None)
    return escape_prose(str(annotation)) if annotation is not None else ""


def signature(obj, drop_self: bool = True) -> str:
    parts = []
    for param in obj.parameters:
        if drop_self and param.name in ("self", "cls"):
            continue
        rendered = param.name
        if param.annotation is not None:
            rendered += f": {param.annotation}"
        if param.default is not None:
            rendered += f" = {param.default}"
        parts.append(rendered)
    returns = getattr(obj, "returns", None)
    suffix = f" -> {returns}" if returns is not None else ""
    return f"{obj.name}({', '.join(parts)}){suffix}"


def decorator_prefix(obj) -> str:
    if "classmethod" in obj.labels:
        return "classmethod "
    if "staticmethod" in obj.labels:
        return "staticmethod "
    return ""


def render_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    if not rows:
        return []
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return lines


def render_sections(obj, anchor: str) -> list[str]:
    docstring = obj.docstring
    if docstring is None:
        return []

    lines: list[str] = []
    for section in docstring.parsed:
        kind = section.kind.value
        if kind not in RENDERED_SECTIONS:
            UNRENDERED_SECTIONS.add(f"{anchor}:{kind}")
            continue

        if kind == "text":
            lines += render_text(section.value)

        elif kind == "parameters":
            rows = [
                [f"`{param.name}`", cell(annotation_of(param)), paragraph(param.description)]
                for param in section.value
            ]
            lines += render_table(["Parameter", "Type", "Description"], rows)

        elif kind in ("returns", "yields", "receives"):
            rows = [
                [
                    f"`{item.name}`" if item.name else "",
                    cell(annotation_of(item)),
                    paragraph(item.description),
                ]
                for item in section.value
            ]
            if not rows:
                continue
            body = " ".join(
                " ".join(part for part in (row[1], row[2]) if part).strip() for row in rows
            )
            label = {"returns": "Returns", "yields": "Yields", "receives": "Receives"}[kind]
            lines += [f"**{label}:** {body}", ""]

        elif kind in ("raises", "warns"):
            rows = [
                [cell(annotation_of(item)), paragraph(item.description)] for item in section.value
            ]
            label = "Raises" if kind == "raises" else "Warns"
            lines += [f"**{label}:**", ""]
            lines += render_table(["Type", "Description"], rows)

        elif kind == "attributes":
            rows = [
                [f"`{item.name}`", cell(annotation_of(item)), paragraph(item.description)]
                for item in section.value
            ]
            lines += render_table(["Attribute", "Type", "Description"], rows)

        elif kind == "examples":
            for _, example in section.value:
                lines += ["```python", example.strip(), "```", ""]

        elif kind == "admonition":
            lines += [f":::note[{paragraph(section.value.title)}]", ""]
            lines += [paragraph(section.value.contents), "", ":::", ""]

        elif kind == "deprecated":
            lines += [":::caution[Deprecated]", "", paragraph(section.value.description), "", ":::", ""]

    return lines


def render_attribute_table(owner) -> list[str]:
    """Attributes of a class or module, documented or not."""
    rows = []
    for name, member in owner.members.items():
        if name.startswith("_") or member.is_alias:
            continue
        if member.kind.value != "attribute" or "property" in member.labels:
            continue
        annotation = member.annotation
        description = paragraph(member.docstring.value) if member.docstring else ""
        rows.append(
            [
                f"`{name}`",
                escape_prose(str(annotation)) if annotation is not None else "",
                description,
            ]
        )
    return render_table(["Attribute", "Type", "Description"], rows)


def render_property(prop, anchor: str) -> list[str]:
    lines = [f'<a id="{anchor}"></a>', "", f"#### {prop.name}", ""]
    docstring = prop.docstring
    if docstring is not None:
        for section in docstring.parsed:
            if section.kind.value == "text":
                lines += [paragraph(section.value), ""]
            elif section.kind.value == "returns":
                for item in section.value:
                    detail = " ".join(
                        part
                        for part in (annotation_of(item), paragraph(item.description))
                        if part
                    )
                    if detail:
                        lines += [f"**Returns:** {detail}", ""]
            else:
                UNRENDERED_SECTIONS.add(f"{anchor}:{section.kind.value}")
    return lines


def render_method(method, anchor: str) -> list[str]:
    lines = [
        f'<a id="{anchor}"></a>',
        "",
        f"#### {method.name}",
        "",
        "```python",
        f"{decorator_prefix(method)}{signature(method)}",
        "```",
        "",
    ]
    lines += render_sections(method, anchor)
    return lines


def documented_members(cls, module_path: str) -> list[tuple[str, object]]:
    members = []
    for name, member in cls.members.items():
        if name.startswith("_"):
            continue
        if "property" in member.labels:
            continue
        if member.is_alias:
            continue
        if member.module is not None and member.module.path != module_path:
            continue
        members.append((name, member))
    return members


def render_class(cls, module_path: str) -> list[str]:
    lines = [f'<a id="{cls.path}"></a>', "", f"## {cls.name}", ""]

    if cls.parameters:
        lines += ["```python", signature(cls), "```", ""]
    else:
        docstring_params = next(
            (s.value for s in (cls.docstring.parsed if cls.docstring else []) if s.kind.value == "parameters"),
            [],
        )
        if docstring_params:
            rendered = ", ".join(
                f"{param.name}: {param.annotation}" if param.annotation else param.name
                for param in docstring_params
            )
            lines += ["```python", f"{cls.name}({rendered})", "```", ""]

    lines += render_sections(cls, cls.path)

    attributes = render_attribute_table(cls)
    if attributes:
        lines += ["### Attributes", ""] + attributes

    properties = [
        (name, member)
        for name, member in cls.members.items()
        if not name.startswith("_") and "property" in member.labels
    ]
    if properties:
        lines += ["### Properties", ""]
        for name, prop in properties:
            lines += render_property(prop, f"{cls.path}.{name}")

    methods = [
        (name, member)
        for name, member in documented_members(cls, module_path)
        if member.kind.value == "function"
    ]
    if methods:
        lines += ["### Methods", ""]
        for name, method in methods:
            lines += render_method(method, f"{cls.path}.{name}")

    return lines


def render_module_page(module, module_index: int) -> str:
    title = module.path
    lines = [
        "---",
        f"title: {title}",
        f"slug: api/{title}",
        f"description: API reference for the {title} module.",
        "sidebar:",
        f"  order: {module_index + 1}",
        "---",
        "",
    ]

    docstring = module.docstring
    if docstring is not None and docstring.value.strip():
        lines += [paragraph(docstring.value), ""]

    functions = []
    classes = []
    attributes = []
    for name, member in module.members.items():
        if name.startswith("_") or member.is_alias:
            continue
        if member.module is not None and member.module.path != module.path:
            continue
        if member.kind.value == "module":
            continue
        if member.kind.value == "function":
            functions.append((name, member))
        elif member.kind.value == "class":
            classes.append((name, member))
        elif member.kind.value == "attribute":
            if member.docstring is not None:
                attributes.append((name, member))
            else:
                SKIPPED_MEMBERS.add(f"{module.path}.{name}")
        else:
            SKIPPED_MEMBERS.add(f"{module.path}.{name}")

    if attributes:
        lines += ["## Attributes", ""]
        rows = [
            [
                f"`{name}`",
                escape_prose(str(member.annotation)) if member.annotation is not None else "",
                paragraph(member.docstring.value),
            ]
            for name, member in attributes
        ]
        lines += render_table(["Attribute", "Type", "Description"], rows)

    if functions:
        lines += ["## Functions", ""]
        for name, function in functions:
            anchor = f"{module.path}.{name}"
            lines += [
                f'<a id="{anchor}"></a>',
                "",
                f"### {name}",
                "",
                "```python",
                f"{decorator_prefix(function)}{signature(function)}",
                "```",
                "",
            ]
            lines += render_sections(function, anchor)

    if classes:
        lines += ["## Classes", ""]
        for _, cls in classes:
            lines += render_class(cls, module.path)

    return "\n".join(lines).rstrip() + "\n"


def render_index_page(modules) -> str:
    lines = [
        "---",
        "title: API Reference",
        "description: The pyfor public API, module by module.",
        "sidebar:",
        "  order: 0",
        "---",
        "",
        "The reference below is generated from the docstrings in `pyfor/`, one page per module.",
        "",
        "| Module | Contents |",
        "| --- | --- |",
    ]
    for _, module in modules:
        classes = sum(
            1
            for member in module.members.values()
            if not member.is_alias and member.kind.value == "class" and not member.name.startswith("_")
        )
        functions = sum(
            1
            for member in module.members.values()
            if not member.is_alias and member.kind.value == "function" and not member.name.startswith("_")
        )
        parts = []
        if classes:
            parts.append(f"{classes} class{'es' if classes != 1 else ''}")
        if functions:
            parts.append(f"{functions} function{'s' if functions != 1 else ''}")
        lines.append(f"| [`{module.path}`](/pyfor/api/{module.path}/) | {', '.join(parts) or '—'} |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    logging.getLogger("griffe").setLevel(logging.ERROR)

    package = griffe.load(
        PACKAGE,
        search_paths=[str(REPO_ROOT)],
        docstring_parser=Parser.sphinx,
    )

    modules = []
    for name in MODULES:
        module = package.modules.get(name)
        if module is None:
            print(f"error: {PACKAGE}.{name} not found", file=sys.stderr)
            return 1
        modules.append((name, module))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for stale in OUT_DIR.glob("*.md"):
        stale.unlink()

    (OUT_DIR / "index.md").write_text(render_index_page(modules), encoding="utf-8")
    for index, (_, module) in enumerate(modules):
        (OUT_DIR / f"{module.path}.md").write_text(
            render_module_page(module, index), encoding="utf-8"
        )

    print(f"wrote {len(modules) + 1} pages to {OUT_DIR.relative_to(REPO_ROOT)}")
    if UNRENDERED_SECTIONS:
        print("unrendered docstring sections:")
        for entry in sorted(UNRENDERED_SECTIONS):
            print(f"  {entry}")
    if SKIPPED_MEMBERS:
        print("members with no documented annotation, skipped:")
        for entry in sorted(SKIPPED_MEMBERS):
            print(f"  {entry}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
