This repository contains a few scripts to setup iGrOw input, read back the
output, and run a few analysis steps.

* igrow_utils.py: write/read iGrOW CSV tables, help with setting up randomly
  generated input.
* analytical.py: contains analytical functions to compute "cell drain
  resistance".
* postprocess.py: utilities to postprocess the iGrOw results.

To run:

* setup-cases.py
* analyze-cases.py
  
As these scripts import from the three Python modules in the first list, make
this directory your working directory.

To just analyze some iGrOw output that hasn't been created with the
``setup-cases`` script, check:

* analysis-example.py
