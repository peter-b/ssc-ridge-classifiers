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

enum _ModelType {
  MODEL_TYPE_GA0,
  MODEL_TYPE_FA,
  MODEL_TYPE_LOGN,
};

typedef enum _ModelType ModelType;

typedef struct _Model Model;
typedef struct _RidgeClass RidgeClass;

struct _Model {
  ModelType type;
  union {
    struct {
      double alpha, gamma, n;
    } ga0;
    struct {
      double L, M, mu;
    } fa;
    struct {
      double sigma, mu;
    } logn;
  } params;
};


struct _RidgeClass {
  double log_weight;
  Model brightness;
  Model strength;
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
"Please report bugs to %s.\n",
name, PACKAGE_BUGREPORT);
  exit (status);
}

static double
key_file_get_double (GKeyFile *key_file, const char *group_name,
                     const char *key)
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
key_file_get_double_prefix (GKeyFile *key_file, const char *group_name,
                            const char *prefix, const char *suffix)
{
  gchar *key = g_strdup_printf ("%s.%s", prefix, suffix);
  double result = key_file_get_double (key_file, group_name, key);
  g_free (key);
  return result;
}

static Model
key_file_get_model (GKeyFile *key_file, const gchar *group_name,
                    const gchar *prefix)
{
  Model m;
  /* Figure out what type of model this is */
  gchar *key;
  gchar *type;
  key = g_strdup_printf ("%s.type", prefix);
  type = g_key_file_get_string (key_file, group_name, key, NULL);
  if (type == NULL || strcmp (type, "FA") == 0) {
    m.type = MODEL_TYPE_FA;
  } else if (strcmp (type, "GA0") == 0) {
    m.type = MODEL_TYPE_GA0;
  } else if (strcmp (type, "LogN") == 0) {
    m.type = MODEL_TYPE_LOGN;
  } else {
    fprintf (stderr, "ERROR: Invalid model type: %s\n", type);
    exit (4);
  }
  g_free (type);
  g_free (key);

  /* Now read params accordingly */
  switch (m.type) {
  case MODEL_TYPE_FA:
    m.params.fa.L = key_file_get_double_prefix (key_file, group_name, prefix, "L");
    m.params.fa.M = key_file_get_double_prefix (key_file, group_name, prefix, "M");
    m.params.fa.mu = key_file_get_double_prefix (key_file, group_name, prefix, "mu");
    break;
  case MODEL_TYPE_GA0:
    m.params.ga0.alpha = key_file_get_double_prefix (key_file, group_name, prefix, "alpha");
    m.params.ga0.gamma = key_file_get_double_prefix (key_file, group_name, prefix, "gamma");
    m.params.ga0.n = key_file_get_double_prefix (key_file, group_name, prefix, "n");
    break;
  case MODEL_TYPE_LOGN:
    m.params.logn.sigma = key_file_get_double_prefix (key_file, group_name, prefix, "sigma");
    m.params.logn.mu = key_file_get_double_prefix (key_file, group_name, prefix, "mu");
    break;
  default:
    g_assert_not_reached ();
  }

  return m;
}

static RidgeClass
key_file_get_class (GKeyFile *key_file, const gchar *group_name)
{
  RidgeClass c;
  c.log_weight = log (key_file_get_double (key_file, group_name, "weight"));
  c.brightness = key_file_get_model (key_file, group_name, "brightness");
  c.strength = key_file_get_model (key_file, group_name, "strength");
  return c;
}

static RidgeClass *
key_file_get_classes (GKeyFile *key_file, gsize *num_classes)
{
  gsize N;
  RidgeClass *classes;
  gchar **groups = g_key_file_get_groups (key_file, &N);
  classes = g_new0 (RidgeClass, N);

  for (int i = 0; i < N; ++i) {
    classes[i] = key_file_get_class (key_file, groups[i]);
  }

  g_assert (num_classes);
  *num_classes = N;
  return classes;
}

static double
lognormpdf (float x, double mu, double sigma)
{
  return gsl_ran_gaussian_pdf (log(x) - mu, sigma) / x;
}

static double
fisherpdf (float x, double L, double M, double mu)
{
  return (2*x)/(mu*mu) * gsl_ran_fdist_pdf ((x*x)/(mu*mu), 2*L, 2*M);
}

static double
ga0pdf (float x, double alpha, double gamma, double n)
{
  return fisherpdf (x, n, -alpha, sqrt (-gamma / alpha));
}

static double
model_pdf (Model m, double x)
{
  switch (m.type) {
  case MODEL_TYPE_GA0:
    return ga0pdf (x,
                   m.params.ga0.alpha,
                   m.params.ga0.gamma,
                   m.params.ga0.n);
  case MODEL_TYPE_FA:
    return fisherpdf (x,
                      m.params.fa.L,
                      m.params.fa.M,
                      m.params.fa.mu);
  case MODEL_TYPE_LOGN:
    return lognormpdf (x,
                       m.params.logn.mu,
                       m.params.logn.sigma);
  default:
    g_assert_not_reached ();
  }
}

static double
ridge_class_log_likelihood (RidgeClass c, RioPoint *p)
{
  return (log (model_pdf (c.strength, powf(p->strength, 0.25)))
          + log (model_pdf (c.brightness, p->brightness)));
}

static double
ridge_class_point_log_likelihood (RidgeClass c, RioPoint *p)
{
  return c.log_weight + ridge_class_log_likelihood (c, p);
}

static double
ridge_class_segment_log_likelihood (RidgeClass c, RioSegment *s)
{
  return (c.log_weight
          + ridge_class_log_likelihood (c, rio_segment_get_start (s))
          + ridge_class_log_likelihood (c, rio_segment_get_end (s)));
}

static double
ridge_class_line_log_likelihood (RidgeClass c, RioLine *l)
{
  double sum = c.log_weight;
  for (int i = 0; i < rio_line_get_length (l); i++) {
    sum += ridge_class_log_likelihood (c, rio_line_get_point (l, i));
  }
  return sum;
}

int
main (int argc, char **argv)
{
  int c, N;
  char *fitfile = NULL;
  char *infile = NULL;
  char *outfile = NULL;
  uint8_t *classification = NULL;
  float *likelihood = NULL;
  RioData *data = NULL;
  GKeyFile *modelkeyfile = NULL;
  GError *err = NULL;
  RidgeClass *classes;
  size_t num_classes;

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

  /* Attempt to load model file */
  modelkeyfile = g_key_file_new ();
  if (!g_key_file_load_from_file (modelkeyfile, fitfile,
                                  G_KEY_FILE_NONE, &err)) {
    fprintf (stderr, "ERROR: Could not load model from %s: %s\n",
             fitfile, err->message);
    exit (4);
  }

  /* Populate models */
  classes = key_file_get_classes (modelkeyfile, &num_classes);
  if (num_classes != 2) {
    fprintf (stderr, "ERROR: Only two classes currently supported.");
    exit (4);
  }

  N = rio_data_get_num_entries (data);
  classification = g_malloc0 (N * sizeof (uint8_t));
  likelihood = g_malloc0 (N * sizeof(float));

  /* Classify */
  for (int i = 0; i < N; i++) {
    double log_ratio = 0;
    RioPoint *p;
    RioSegment *s;
    RioLine *l;
    switch (rio_data_get_type (data)) {
    case RIO_DATA_POINTS:
      p = rio_data_get_point (data, i);
      log_ratio = (ridge_class_point_log_likelihood (classes[1], p)
               - ridge_class_point_log_likelihood (classes[0], p));
      break;
    case RIO_DATA_SEGMENTS:
      s = rio_data_get_segment (data, i);
      log_ratio = (ridge_class_segment_log_likelihood (classes[1], s)
               - ridge_class_segment_log_likelihood (classes[0], s));
      break;
    case RIO_DATA_LINES:
      l = rio_data_get_line (data, i);
      log_ratio = (ridge_class_line_log_likelihood (classes[1], l)
               - ridge_class_line_log_likelihood (classes[0], l));
      break;
    default:
      g_assert_not_reached ();
    }

    classification[i] = (log_ratio > 0);
    likelihood[i] = expf (log_ratio);
  }

  /* Save results */
  rio_data_set_metadata (data, RIO_KEY_IMAGE_CLASSIFICATION,
                         (char *) classification, N * sizeof (uint8_t));

  /* Swab likelihood values */
  for (int i = 0; i < N; i++)
    likelihood[i] = rio_htonf (likelihood[i]);

  rio_data_set_metadata (data, RIO_KEY_IMAGE_CLASS_LIKELIHOOD,
                         (char *) likelihood, N * sizeof (float));

  if (!rio_data_to_file (data, outfile)) {
    const char *msg = errno ? strerror (errno) : "Unexpected error";
    fprintf (stderr, "ERROR: Could not save ridge data to %s: %s\n",
             outfile, msg);
    exit (5);
  }

  rio_data_destroy (data);
  g_free (classes);
  g_free (classification);
  g_free (likelihood);
  g_key_file_free (modelkeyfile);
  return 0;
}
