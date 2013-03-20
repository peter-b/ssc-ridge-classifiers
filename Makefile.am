bin_PROGRAMS = ridge-classarea ridge-classbayes

ridge_classarea_SOURCES = ridge-classarea.c
ridge_classbayes_SOURCES = ridge-classbayes.c

CFLAGS = -g -Wall -pedantic $(RIDGETOOL_CFLAGS) $(GSL_CFLAGS) $(GLIB_CFLAGS)
LDADD = $(RIDGETOOL_LIBS) $(GSL_LIBS) $(GLIB_LIBS)
