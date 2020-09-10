#!/usr/bin/env python3
import argparse
import json
import logging
import os
import re
import sys
import traceback
from collections import defaultdict
from functools import partial
from pathlib import Path

from robot.errors import DataError
from robot.libdocpkg import LibraryDocumentation, htmlwriter
from robot.utils import normalize, unic


class Peekable:
    """Convert an iterable into an iterator that can be peeked ahead,
    without consuming the peeked values.
    """

    SENTINEL = object()

    def __init__(self, iterable):
        self._iter = iter(iterable)
        self._peek = []

    def __iter__(self):
        return self

    def __next__(self):
        if self._peek:
            return self._peek.pop(0)

        return next(self._iter)

    @property
    def sentinel(self):
        return self.SENTINEL

    def peek(self, ahead=1):
        ahead = int(ahead)
        if ahead > len(self._peek):
            peeked = []
            for idx, value in enumerate(self, 1):
                peeked.append(value)
                if idx >= ahead:
                    break
            self._peek = peeked

        if ahead <= len(self._peek):
            return self._peek[ahead - 1]

        return self.sentinel

    def has_next(self):
        return self.peek() is not self.sentinel

    def next(self):
        return next(self)


class Docstring:
    """Parser class for converting google-style docstrings into dictionaries."""

    REGEX_DIRECTIVE = re.compile(r"\.\. \S+::")
    REGEX_SECTION = re.compile(r"^(\s|\w)+:\s*$")
    REGEX_TYPED_ARG = re.compile(r"\s*(.+?)\s*\(\s*(.*[^\s]+)\s*\)")
    REGEX_SINGLE_COLON = re.compile(r"(?<!:):(?!:)")
    REGEX_XREF = re.compile(r"(:(?:[a-zA-Z0-9]+[\-_+:.])*[a-zA-Z0-9]+:`.+?`)")
    REGEX_NAME = re.compile(
        r"^\s*(:(?P<role>\w+):`(?P<name>[a-zA-Z0-9_.-]+)`|"
        r" (?P<name2>[a-zA-Z0-9_.-]+))\s*",
        re.X,
    )

    def __init__(self):
        self._lines = None
        self._is_in_section = False
        self._section_indent = 0
        self._sections = {
            "args": self._parse_arguments,
            "arguments": self._parse_arguments,
            "parameters": self._parse_arguments,
            "other parameters": self._parse_arguments,
            "keyword args": self._parse_arguments,
            "keyword arguments": self._parse_arguments,
            "attributes": self._parse_attributes,
            "methods": self._parse_methods,
            "raises": self._parse_raises,
            "return": self._parse_returns,
            "returns": self._parse_returns,
            "yield": self._parse_returns,
            "yields": self._parse_returns,
            "example": partial(self._parse_block, "examples"),
            "examples": partial(self._parse_block, "examples"),
            "reference": partial(self._parse_block, "references"),
            "references": partial(self._parse_block, "references"),
            "attention": partial(self._parse_info, "attention"),
            "caution": partial(self._parse_info, "caution"),
            "danger": partial(self._parse_info, "danger"),
            "error": partial(self._parse_info, "error"),
            "hint": partial(self._parse_info, "hint"),
            "important": partial(self._parse_info, "important"),
            "note": partial(self._parse_info, "note"),
            "notes": partial(self._parse_info, "note"),
            "see also": partial(self._parse_info, "seealso"),
            "tip": partial(self._parse_info, "tip"),
            "todo": partial(self._parse_info, "todo"),
            "warning": partial(self._parse_info, "warning"),
            "warnings": partial(self._parse_info, "warning"),
        }

    def parse(self, docstring):
        """Parse docstring into a dictionary of different sections."""
        self._lines = Peekable(line.rstrip() for line in docstring.splitlines())
        self._read_empty()

        body = []
        sections = defaultdict(list)

        while self._lines.has_next():
            if not self._is_section_header():
                if not sections:
                    lines = self._read_contiguous() + self._read_empty()
                else:
                    lines = self._read_to_next_section()
                body.extend(lines)
            else:
                try:
                    section = self._read_section_header()
                    self._is_in_section = True
                    self._section_indent = self._get_current_indent()

                    is_directive = self.REGEX_DIRECTIVE.match(section)
                    is_section = section.lower() in self._sections

                    if is_directive or not is_section:
                        lines = [section] + self._read_to_next_section()
                        body.extend(lines)
                    else:
                        name, fields = self._sections[section.lower()]()
                        sections[name].extend(fields)
                finally:
                    self._is_in_section = False
                    self._section_indent = 0

        sections["doc"] = "\n".join(self._strip_empty(body))

        return sections

    def _parse_block(self, name):
        lines = self._read_to_next_section()
        lines = self._dedent(lines)
        lines = self._strip_empty(lines)

        fields = ["\n".join(lines)]
        return name, fields

    def _parse_info(self, name):
        name, fields = self._parse_block(name)
        return "info", [{"name": "", "type": name, "desc": fields[0]}]

    def _parse_arguments(self):
        return "args", self._read_fields()

    def _parse_attributes(self):
        return "attributes", self._read_fields()

    def _parse_methods(self):
        return "methods", self._read_fields(parse_type=False)

    def _parse_returns(self):
        return "returns", self._read_returns_section()

    def _parse_raises(self):
        fields = self._read_fields(parse_type=False, prefer_type=True)

        for field in fields:
            match = self.REGEX_NAME.match(field["type"]).groupdict()
            if match["role"]:
                field["type"] = match["name"]

        return "raises", fields

    def _read_indented_block(self, indent=1):
        """Read lines until current indentation block ends."""
        lines = []
        line = self._lines.peek()

        while not self._is_section_break() and (
            not line or self._is_indented(line, indent)
        ):
            lines.append(next(self._lines))
            line = self._lines.peek()

        return lines

    def _read_contiguous(self):
        """Read lines until break or new section."""
        lines = []

        while (
            self._lines.has_next()
            and self._lines.peek()
            and not self._is_section_header()
        ):
            lines.append(next(self._lines))

        return lines

    def _read_empty(self):
        """Read until the first non-empty line."""
        lines = []
        line = self._lines.peek()

        while self._lines.has_next() and not line:
            lines.append(next(self._lines))
            line = self._lines.peek()

        return lines

    def _read_field(self, parse_type=True, prefer_type=False):
        """Read field line with name, optional type, and description."""
        line = next(self._lines)
        before, _, after = self._partition_colon(line)

        field_name = before
        field_type = ""

        if parse_type:
            match = self.REGEX_TYPED_ARG.match(field_name)
            if match:
                field_name = match.group(1)
                field_type = match.group(2)

        if prefer_type and not field_type:
            field_type, field_name = field_name, field_type

        indent = self._get_indent(line) + 1
        lines = [after] + self._dedent(self._read_indented_block(indent))
        lines = self._strip_empty(lines)

        return {"name": field_name, "type": field_type, "desc": "\n".join(lines)}

    def _read_fields(self, parse_type=True, prefer_type=False):
        """Read section of generic fields."""
        self._read_empty()

        fields = []
        while not self._is_section_break():
            field = self._read_field(parse_type, prefer_type)
            fields.append(field)

        return fields

    def _read_returns_section(self):
        """Read section of return fields."""
        lines = self._read_to_next_section()
        lines = self._dedent(lines)
        lines = self._strip_empty(lines)

        if not lines:
            return []

        field_type = ""
        before, colon, after = self._partition_colon(lines[0])

        if colon:
            field_type = before
            lines = [after] + lines[1:] if after else lines[1:]

        return [{"name": "", "type": field_type, "desc": "\n".join(lines)}]

    def _read_section_header(self):
        """Read next line as section header."""
        section = next(self._lines)

        stripped_section = section.strip(":")
        if stripped_section.lower() in self._sections:
            section = stripped_section

        return section

    def _read_to_next_section(self):
        """Read until the current section ends."""
        self._read_empty()

        lines = []
        while not self._is_section_break():
            lines.append(next(self._lines))

        return lines + self._read_empty()

    def _get_indent(self, line):
        """Get the indent level for the given line."""
        for idx, char in enumerate(line):
            if not char.isspace():
                return idx

        return len(line)

    def _get_current_indent(self, peek_ahead=0):
        """Get the current indent level from the next non-empty line."""
        line = self._lines.peek(peek_ahead + 1)

        while line != self._lines.sentinel:
            if line:
                return self._get_indent(line)
            peek_ahead += 1
            line = self._lines.peek(peek_ahead + 1)

        return 0

    def _get_initial_indent(self, lines):
        """Get the indent level for the first non-empty line."""
        for line in lines:
            if line:
                return self._get_indent(line)
        return 0

    def _get_min_indent(self, lines):
        """Get the smallest shared indent level for a block of lines."""
        min_indent = None
        for line in lines:
            if line:
                indent = self._get_indent(line)
                if min_indent is None:
                    min_indent = indent
                elif indent < min_indent:
                    min_indent = indent

        return min_indent or 0

    def _indent(self, lines, count=4):
        """Add given indent level to a line."""
        return [(" " * int(count)) + line for line in lines]

    def _dedent(self, lines):
        """Remove indentation from a block to the minimum shared level."""
        min_indent = self._get_min_indent(lines)
        return [line[min_indent:] for line in lines]

    def _is_indented(self, line, indent=1):
        """Check if line has at least a certain indent level."""
        for idx, char in enumerate(line):
            if idx >= indent:
                return True
            if not char.isspace():
                return False
        return False

    def _is_section_header(self):
        """Check if docstring is at a section header."""
        section = self._lines.peek().lower()
        match = self.REGEX_SECTION.match(section)

        if match and section.strip(":") in self._sections:
            header_indent = self._get_indent(section)
            section_indent = self._get_current_indent(peek_ahead=1)
            return section_indent > header_indent

        return False

    def _is_section_break(self):
        """Check if docstring is at the end of a section."""
        line = self._lines.peek()
        return (
            not self._lines.has_next()
            or self._is_section_header()
            or (
                self._is_in_section
                and line
                and not self._is_indented(line, self._section_indent)
            )
        )

    def _partition_colon(self, line):
        """Partition line at first individual colon."""
        before, after = [], []
        colon = ""

        found_colon = False
        for idx, source in enumerate(self.REGEX_XREF.split(line)):
            if found_colon:
                after.append(source)
                continue

            match = self.REGEX_SINGLE_COLON.search(source)
            if (idx % 2) == 0 and match:
                found_colon = True
                colon = source[match.start() : match.end()]
                before.append(source[: match.start()])
                after.append(source[match.end() :])
            else:
                before.append(source)

        before = "".join(before).strip()
        after = "".join(after).strip()

        return before, colon, after

    def _strip_empty(self, lines):
        """Remove empty lines at the beginning and end of a block."""
        if not lines:
            return []

        start = None
        for idx, line in enumerate(lines):
            if line:
                start = idx
                break

        if start is None:
            return []

        end = None
        for idx, line in reversed(list(enumerate(lines))):
            if line:
                end = idx
                break

        return lines[start : end + 1]


class NullFormatter:
    def html(self, doc, *args, **kwargs):
        return doc


class Docgen:
    INIT_FILES = ("__init__.robot", "__init__.txt")
    EXTENSIONS = (".robot", ".resource", ".txt")
    BLACKLIST = ("__pycache__",)

    def convert_all(self, search_dir):
        logging.info("Searching for libraries in '%s'", search_dir)
        files = self.find_library_files(search_dir)

        if not files:
            raise ValueError("No keyword files found")

        root = Path(os.path.commonprefix(files)).parent

        output = []
        errors = set()
        kwcount = 0
        for path in files:
            try:
                converted = self.convert_file(root, path)
                output.append(converted)
                kwcount += len(converted["keywords"])
            except Exception as err:
                errors.add(path)
                logging.error(str(err).split("\n")[0])
                if logging.getLogger().isEnabledFor(logging.DEBUG):
                    traceback.print_exc()

        print(f"Keyword count: {kwcount}")

        if errors:
            logging.error(
                "Failed to convert the following libraries: %s",
                ",".join(str(e) for e in sorted(errors)),
            )

        return len(errors), output

    def convert_file(self, root, path):
        logging.debug("Converting '%s'", path)
        libdoc = LibraryDocumentation(str(path), doc_format="text")

        namespace = str(path.parent.relative_to(root)).replace(os.sep, ".")
        if namespace != ".":
            libdoc.name = namespace + "." + libdoc.name

        scope = normalize(unic(libdoc.scope), ignore="_")
        libdoc.scope = {
            "testcase": "Task",
            "testsuite": "Suite",
            "global": "Global",
        }.get(scope, "")

        docs = {}
        for keyword in libdoc.keywords:
            doc = Docstring().parse(keyword.doc)
            defaults = (arg.partition("=")[2] for arg in keyword.args)
            for arg, default in zip(doc["args"], defaults):
                arg["default"] = default
            docs[keyword.name] = doc

        writer = htmlwriter.JsonConverter(NullFormatter())
        output = writer.convert(libdoc)

        for keyword in output["keywords"]:
            keyword.update(docs[keyword["name"]])

        return output

    def find_library_files(self, root):
        files, stack = set(), [Path(root)]

        while stack:
            path = stack.pop(0)
            if path.name in self.BLACKLIST:
                continue

            if path.is_dir():
                if self.is_module_library(path):
                    files.add(path)
                else:
                    for child in path.iterdir():
                        stack.append(child)
            elif self.is_library_file(path):
                files.add(path)

        return [Path(f) for f in files]

    def is_module_library(self, path):
        return any((path / init).is_file() for init in self.INIT_FILES) and bool(
            LibraryDocumentation(str(path)).keywords
        )

    def is_library_file(self, path):
        return path.suffix == ".py" and path.name not in self.INIT_FILES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("search", help="Search path", type=Path)
    parser.add_argument("output", help="Output JSON path", type=Path)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        stream=sys.stdout,
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
    )

    errors, converted = Docgen().convert_all(args.search)

    if errors:
        sys.exit(errors)

    with open(args.output, "w") as fd:
        json.dump(converted, fd, indent=2)


if __name__ == "__main__":
    main()
