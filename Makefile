#SCRIPT=segment
#SCRIPT=cls
SCRIPT=depth

run:
	-sbatch $(SCRIPT).sh
