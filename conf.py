# -*- coding: utf-8 -*-

import os
import sys

sys.path.insert(0, os.path.abspath("extensions"))

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.pngmath",
    "sphinx.ext.ifconfig",
    "epub2",
    "mobi",
    "autoimage",
    "code_example",
    "sphinxcontrib.napoleon",
]

todo_include_todos = True
templates_path = ["_templates"]
source_suffix = ".rst"
master_doc = "index"
exclude_patterns = []
add_function_parentheses = True
# add_module_names = True
# A list of ignored prefixes for module index sorting.
# modindex_common_prefix = []

project = "Reinforcement Learning Dogfight"
copyright = "2022, Rana Riaz"

version = ""
release = ""
