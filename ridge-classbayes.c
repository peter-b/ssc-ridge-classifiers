/*
 * Surrey Space Centre ridge tools for SAR data processing
 * Copyright (C) 2011-2012  Peter Brett <p.brett@surrey.ac.uk>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "config.h"

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#include <math.h>

#include <ridgeio.h>

#include <glib.h>
#include <gsl/gsl_randist.h>

#define GETOPT_OPTIONS "h"

typedef struct _RidgeModel RidgeModel;

struct _RidgeModel {
  double weight;
  double s_sigma, s_mu;
  double b_l, b_m, b_mu;
};

static void
usage (char *name, int status)
{
  printf (
"Usage: %s OPTION... FITFILE INFILE [OUTFILE]\n"
"\n"
"Classify ridge lines based on a naive Bayesian model.\n"
"\n"
"  -h              Display this message and exit\n"
"\n"
"Loads ridges from INFILE, and classifies them according to the\n"
"parameters in FITFILE. If OUTFILE is not specified, results are\n"
"written back to INFILE.\n"
"\n"
"If INFILE already contains classification data, the existing\n"
"classification is treated as a reference classification, and\n"
"misclassification statistics are generated on standard output.\n"
"In this case, a new classification is never written to INFILE.\n"
"\n"
"Please report bugs to %s.\n",
name, PACKAGE_BUGREPORT);
  exit (status);
}

static double
key_file_get_double (GKeyFile *key_file, char *group_name, char *key)
{
  GError *err = NULL;
  double result = g_key_file_get_double (key_file, group_name, key, &err);
  if (err != NULL) {
    fprintf (stderr, "ERROR: Could not load model: %s\n", err->message);
    exit (4);
  }
  return result;
}

static double
lognormpdf (float x, double mu, double sigma)
{
  return gsl_ran_gaussian_pdf (log(x) - mu, sigma) / x;
}

static double
fisherpdf (float x, double L, double M, double mu)
{
  return (2*x)/(mu*mu) * gsl_ran_fdist_pdf (x / mu, 2*L, 2*M);
}

static double
log_likelihood (RidgeModel m, RioPoint *p)
{
  return (log (lognormpdf (p->strength, m.s_mu, m.s_sigma))
          + log (fisherpdf (p->brightness, m.b_l, m.b_m, m.b_mu)));
}

static double
point_log_likelihood (RidgeModel m, RioPoint *p)
{
  return m.weight + log_likelihood (m, p);
}

static double
segment_log_likelihood (RidgeModel m, RioSegment *s)
{
  return (m.weight
          + log_likelihood (m, rio_segment_get_start (s))
          + log_likelihood (m, rio_segment_get_end (s)));
}

static double
line_log_likelihood (RidgeModel m, RioLine *l)
{
  double sum = m.weight;
  for (int i = 0; i < rio_line_get_length (l); i++) {
    sum += log_likelihood (m, rio_line_get_point (l, i));
  }
  return sum;
}

int
main (int argc, char **argv)
{
  int c;
  char *fitfile = NULL;
  char *infile = NULL;
  char *outfile = NULL;
  uint8_t *reference = NULL;
  uint8_t *classification = NULL;
  size_t classification_size = 0;
  RioData *data = NULL;
  GKeyFile *modelkeyfile = NULL;
  GError *err = NULL;
  RidgeModel modelA, modelB;

  /* Parse command-line arguments */
  while ((c = getopt (argc, argv, GETOPT_OPTIONS)) != -1) {
    switch (c) {
   case 'h':
      usage (argv[0], 0);
      break;
    case '?':
      if ((optopt != ':') && (strchr (GETOPT_OPTIONS, optopt) != NULL)) {
        fprintf (stderr, "ERROR: -%c option requires an argument.\n\n", optopt);
      } else if (isprint (optopt)) {
        fprintf (stderr, "ERROR: Unknown option -%c.\n\n", optopt);
      } else {
        fprintf (stderr, "ERROR: Unknown option character '\\x%x'.\n\n",
                 optopt);
      }
      usage (argv[0], 1);
    default:
      abort ();
    }
  }

  /* Get input and output filenames */
  if (argc - optind < 2) {
    fprintf (stderr, "ERROR: You must specify a model file and input file.\n\n");
    usage (argv[0], 1);
  }
  fitfile = argv[optind];
  infile = argv[optind+1];
  if (argc - optind > 2) {
    outfile = argv[optind+2];
  } else {
    outfile = infile;
  }

  /* Attempt to load input file */
  data = rio_data_from_file (infile);
  if (data == NULL) {
    const char *msg = errno ? strerror (errno) : "Unexpected error";
    fprintf (stderr, "ERROR: Could not load ridge data from %s: %s\n",
             infile, msg);
    exit (2);
  }

  /* Check metadata & create classification if necessary */
  reference = (uint8_t *) rio_data_get_metadata (data,
                                                 RIO_KEY_IMAGE_CLASSIFICATION,
                                                 &classification_size);
  if (reference != NULL
      && classification_size != (sizeof (uint8_t)
                                 * rio_data_get_num_entries (data))) {
    fprintf (stderr ,"ERROR: %s contains invalid classification metadata\n",
             infile);
    exit (3);
  }

  classification_size = sizeof (uint8_t) * rio_data_get_num_entries (data);
  classification = malloc (classification_size);
  memset (classification, 0, classification_size);


  /* Attempt to load model file */
  modelkeyfile = g_key_file_new ();
  if (!g_key_file_load_from_file (modelkeyfile, fitfile,
                                  G_KEY_FILE_NONE, &err)) {
    fprintf (stderr, "ERROR: Could not load model from %s: %s\n",
             fitfile, err->message);
    exit (4);
  }

  /* Populate models */
  modelA.weight = key_file_get_double (modelkeyfile, "A", "weight");
  modelA.s_sigma = key_file_get_double (modelkeyfile, "A", "strength.sigma");
  modelA.s_mu = key_file_get_double (modelkeyfile, "A", "strength.mu");
  modelA.b_l = key_file_get_double (modelkeyfile, "A", "brightness.l");
  modelA.b_m = key_file_get_double (modelkeyfile, "A", "brightness.m");
  modelA.b_mu = key_file_get_double (modelkeyfile, "A", "brightness.mu");

  modelB.weight = key_file_get_double (modelkeyfile, "B", "weight");
  modelB.s_sigma = key_file_get_double (modelkeyfile, "B", "strength.sigma");
  modelB.s_mu = key_file_get_double (modelkeyfile, "B", "strength.mu");
  modelB.b_l = key_file_get_double (modelkeyfile, "B", "brightness.l");
  modelB.b_m = key_file_get_double (modelkeyfile, "B", "brightness.m");
  modelB.b_mu = key_file_get_double (modelkeyfile, "B", "brightness.mu");

  /* Classify */
  int false_alarms = 0;
  int misses = 0;
  int N_A = 0;
  int N_B = 0;

  for (int i = 0; i < rio_data_get_num_entries (data); i++) {
    double klass = 0;
    RioPoint *p;
    RioSegment *s;
    RioLine *l;
    switch (rio_data_get_type (data)) {
    case RIO_DATA_POINTS:
      p = rio_data_get_point (data, i);
      klass = (point_log_likelihood (modelB, p)
               - point_log_likelihood (modelA, p));
      break;
    case RIO_DATA_SEGMENTS:
      s = rio_data_get_segment (data, i);
      klass = (segment_log_likelihood (modelB, s)
               - segment_log_likelihood (modelA, s));
      break;
    case RIO_DATA_LINES:
      l = rio_data_get_line (data, i);
      klass = (line_log_likelihood (modelB, l)
               - line_log_likelihood (modelA, l));
      break;
    default:
      abort ();
    }

    classification[i] = (klass > 0);
    N_A += classification[i];
    N_B += !classification[i];

    if (reference != NULL) {
      misses += (reference[i] && !classification[i]);
      false_alarms += (!reference[i] && classification[i]);
    }
  }

  /* Save results */
  if ((reference == NULL) || (infile != outfile)) {
    rio_data_take_metadata (data, RIO_KEY_IMAGE_CLASSIFICATION,
                            (char *) classification,
                            classification_size);

    if (!rio_data_to_file (data, outfile)) {
      const char *msg = errno ? strerror (errno) : "Unexpected error";
      fprintf (stderr, "ERROR: Could not save ridge data to %s: %s\n",
               infile, msg);
      exit (5);
    }
  }

  printf ("%i classified (A: %i, B: %i)\n", N_A + N_B, N_A, N_B);
  if (reference) {
    printf ("%i (%f %%) misses, %i (%f %%) false alarms\n",
            misses, (double) 100 * misses / (N_A + N_B),
            false_alarms, (double) 100 * false_alarms / (N_A + N_B));
  }

  return 0;
}
