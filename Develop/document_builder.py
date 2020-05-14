import os
import sphinx.cmd.make_mode as sphinx_build

OUT_DIR = "../ApiSphinxDocumentation/source/"  # here you have your conf.py etc
build_output = os.path.join("../ApiSphinxDocumentation/", "build/")

# build HTML (same as `make html`)
build_html_args = ["html", OUT_DIR, build_output]
sphinx_build.run_make_mode(args=build_html_args)

# build PDF latex (same as `make latexpdf`)
build_pdf_args = ["latexpdf", OUT_DIR, build_output]
sphinx_build.run_make_mode(args=build_pdf_args)