AR     = ar rcv
RANLIB = ranlib 

HEADERS = blas.h blas.h blasp.h
FILES = snrm2.o saxpy.o sdot.o sscal.o 

CFLAGS = $(OPTFLAGS) 
FFLAGS = $(OPTFLAGS)

blas: $(FILES) $(HEADERS)
	$(AR) blas.a $(FILES)  
	$(RANLIB) blas.a

clean:
	- rm -f *.o
	- rm -f *.a
	- rm -f *~

.c.o:
	$(CC) $(CFLAGS) -c $*.c


