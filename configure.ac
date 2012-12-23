#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.63])
AC_INIT([SSC Ridge Classifiers], [0.1], [p.brett@surrey.ac.uk])

AC_CONFIG_SRCDIR([ridge-classarea.c])
AC_CONFIG_AUX_DIR([build-tools])
AC_CONFIG_MACRO_DIR([m4])
AC_CONFIG_HEADERS([config.h])

AM_INIT_AUTOMAKE([foreign])

# Checks for programs
AC_PROG_CC
AC_PROG_CC_C99
PKG_PROG_PKG_CONFIG

# Checks for libraries
PKG_CHECK_MODULES([GSL], [gsl >= 1.13], [],
  AC_MSG_ERROR([GNU Scientific Library 1.13.0 or later is required.]))
PKG_CHECK_MODULES([RIDGETOOL], [libridgetool], [],
  AC_MSG_ERROR([SSC Ridge Tools Library is required.]))
PKG_CHECK_MODULES([GLIB], [glib-2.0 >= 2.16], [],
  AC_MSG_ERROR([GLib 2.16.0 or later is required.]))

AC_CONFIG_FILES([Makefile])
AC_OUTPUT
