#! /usr/bin/python
# -*- coding: utf-8 -*-
import re
import sys
import random

def convert_set(filename, max_examples=None):
    features, classes, examples=get_set_params(filename)
    file_appendix=".set"
    print examples
    if max_examples and int(max_examples)<examples:
        examples=int(max_examples)
        print "set examples to ", examples
        file_appendix="_"+str(examples)+".set"
    with open("converted/"+filename.replace(".src_set", file_appendix), 'w') as dest_set:
        dest_set.write( "%s %s %s\n" % (examples, features, classes)  )
        with open(filename, 'r') as src_set:
            src_lines=src_set.readlines()
            random.shuffle(src_lines)
            src_lines=src_lines[:examples]
            for src_line in src_lines:
                inps, cls=get_line_data(src_line, int(classes))
                write_data(dest_set, inps, cls)
    return

def get_set_params(desc):
    desc_regexp=re.compile("^(?P<features>\d+)f_(?P<classes>\d+)c_(?P<examples>\d+)e\.src_set$")
    parsed_desc=desc_regexp.search(desc)
    return [ int(parsed_desc.group("features")), int(parsed_desc.group("classes")), int(parsed_desc.group("examples")) ]

def get_line_data(str_data, Nclasses):
    tokens=str_data.split("\t")
    cls=expand_class(tokens[0], Nclasses)
    # workaround to get rid of trailing newlines
    inps=tokens[1 : -1]
    # workaround to remove excessive digits
    inps=[ tkn[:9] for tkn in inps]
    return [inps, cls]

def expand_class(cls, Nclasses):
    cls_exp=[str(item) for item in [0]*Nclasses ]
    cls_exp[int(cls)]="1"
    return cls_exp

def write_data(dest_file, inps, cls):
    dest_file.write(" ".join(inps)+"\n" )
    dest_file.write(" ".join(cls) +"\n" )
    
convert_set(sys.argv[1], *sys.argv[2:])
