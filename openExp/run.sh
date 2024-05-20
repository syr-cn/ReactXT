#!/bin/bash
{
	name="processed";
    root="data";
	mkdir -p $root/$name

	python process.py \
        --root $root \
        --name $name \
        >$root/$name/process.out 2>$root/$name/process.err;
}