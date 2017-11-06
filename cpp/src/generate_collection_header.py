#!/usr/bin/env python3

import sys
import argparse
import os.path
import re


def generate_include_header(file, includes):
    guard = re.sub(r"\W", "_", os.path.basename(file)).upper()
    frame = ("#ifndef {guard}\n" +
             "#define {guard}\n" +
             "{includes}" +
             "#endif\n")
    incs = ""
    for f in includes:
        incs += '#include "{}"\n'.format(f)
    return frame.format(guard=guard, includes=incs)

def main():
    parser = argparse.ArgumentParser(description="Generate a header file including multiple files.")
    parser.add_argument("-o", "--output", help="Output File")
    parser.add_argument("includes", metavar="F",
                        help="List of files to include",
                        nargs="+")
    args = parser.parse_args()

    code = generate_include_header(args.output, args.includes)

    with open(args.output, mode='w') as f:
        f.write(code)

if __name__ == "__main__":
    main()
