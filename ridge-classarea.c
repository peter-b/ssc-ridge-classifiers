/*
 * Surrey Space Centre ridge tools for SAR data processing
 * Copyright (C) 2012  Peter Brett <p.brett@surrey.ac.uk>
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

#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <errno.h>

#include <glib.h>
#include <gsl/gsl_sf_bessel.h>
#include <gsl/gsl_sf_gamma.h>

#include <ridgeutil.h>
#include <ridgeio.h>

#define GETOPT_OPTIONS "m:t:s:p:j:h"

enum {
  MODE_AREA,
  MODE_AREA_LENGTH,
};

static double DEFAULT_THRESHOLD = 0.12;
static double DEFAULT_PARAM_LAMBDA_X = 1.0;
static double DEFAULT_PARAM_LAMBDA_Y = 1.0;
static double DEFAULT_PARAM_K = 4;
static double DEFAULT_PARAM_M = 5;

typedef struct _SubprocessData SubprocessData;

struct _SubprocessData {
  RioData *lines;
  gchar *classification;
  float *likelihood;

  int mode;
  double threshold;
  double param_lambda_x;
  double param_lambda_y;
  double param_k;
  double param_m;
};

void
usage (const char *name, int status)
{
  printf (
"Usage: %s OPTION... IN_FILE [OUT_FILE]\n"
"\n"
"Classify ridge lines using internal area geometry.\n"
"\n"
"  -m MODE         Classification mode (A or AL) [default A]\n"
"  -t THRESHOLD    Model likelihood threshold [default 0.005]\n"
"  -s X:Y          Scale factors in ground range/azimuth [default 1]\n"
"  -p K:M          Building Gamma model parameters [default 4,5]\n"
"  -j THREADS      Number of multiprocessing threads to use [default 1]\n"
"  -h              Display this message and exit\n"
"\n"
"Reads ridge line data generated using 'ridgetool' from IN_FILE, and\n"
"classifies it using a likelihood threshold classification based on\n"
"analytical models derived from stereotypical scene geometry.\n"
"Classification results are written to OUT_FILE, if specified;\n"
"otherwise, IN_FILE is updated.\n"
"\n"
"Please report bugs to %s.\n",
name, PACKAGE_BUGREPORT);
  exit (status);
}

/* Calculate total length of line */
static double
stat_line_length (RioLine *l)
{
  int M = rio_line_get_length (l);
  if (M <= 1) return 0; /* Sanity */

  double len = 0;
  RioPoint *p;
  double x0, x1, y0, y1, dx, dy;

  /* First point */
  p = rio_line_get_point (l, 0);
  rio_point_get_subpixel (p, &x0, &y0);

  int i = 1;
  while (i < M) {
    p = rio_line_get_point (l, i);
    rio_point_get_subpixel (p, &x1, &y1);

    /* Calculate length of segment */
    dx = x1 - x0;
    dy = y1 - y0;
    len += sqrt (dx * dx + dy * dy);

    /* iterate */
    i++;
    x0 = x1;
    y0 = y1;
  }
  return len;
}

/* Calculate end-to-end distance of line */
static double
stat_line_end_to_end (RioLine *l)
{
  int M = rio_line_get_length (l);
  if (M <= 1) return 0; /* Sanity */

  RioPoint *p;
  double x0, x1, y0, y1, dx, dy;
  p = rio_line_get_point (l, 0);
  rio_point_get_subpixel (p, &x0, &y0);
  p = rio_line_get_point (l, M-1);
  rio_point_get_subpixel (p, &x1, &y1);
  dx = x1 - x0;
  dy = y1 - y0;
  return sqrt (dx*dx + dy*dy);
}

/* Calculate radius of gyration of line */
static double
stat_line_gyration (RioLine *l)
{
  int M = rio_line_get_length (l);
  RioPoint *p;

  double x0, xk, xM, y0, yk, yM;
  /* Get first and last points */
  p = rio_line_get_point (l, 0);
  rio_point_get_subpixel (p, &x0, &y0);
  p =  rio_line_get_point (l, M-1);
  rio_point_get_subpixel (p, &xM, &yM);

  /* Treat start point as origin */
  xM -= x0; yM -= y0;

  double Rg2 = 0;
  for (int k = 0; k < M; k++) {
    p = rio_line_get_point (l, k);
    rio_point_get_subpixel (p, &xk, &yk);

    /* Treat start point as origin */
    xk -= x0; yk -= y0;

    /* Cross product */
    double cp = xk * yM - yk * xM;

    Rg2 += cp * cp;
  }

  /* Normalise */
  double Re = stat_line_end_to_end (l);
  return Rg2 / (M * Re * Re);
}

/* Calculate prior probability of building using area only */
static double
prior_building_area (RioLine *l,
                     double lambda_x, double lambda_y,
                     double k, double m)
{
  double Re = stat_line_end_to_end (l);
  double Rg2 = stat_line_gyration (l);

  /* Estimate of projected area */
  double t = Re * sqrt (3 * Rg2);
  double Ct = lambda_x * lambda_y;
  double tmp = t / Ct / (m*m);
  double p;

  /* Cope with Re = 0 case */
  if (Re <= 0 || t <= 0) return 0;

  p = (2 * pow (tmp, k) * gsl_sf_bessel_K0 (2 * sqrt (tmp)))
    / (t * pow (gsl_sf_gamma (k), 2));

  return p;
}

static double
prior_building_area_length (RioLine *l, double lambda_x,
                            double lambda_y, double k, double m)
{
  double Re = stat_line_end_to_end (l);
  double Rg2 = stat_line_gyration (l);
  double L = stat_line_length (l);

  /* Estimate of projected area */
  double t = Re * sqrt (3 * Rg2);
  double Ct = lambda_x * lambda_y;
  double tmp = L*L - 4*t;

  /* Iverson bracket */
  if (t <= 0 || L <= 0 || tmp <= 0) return 0;

  return 2 / sqrt (Ct * tmp)
    * pow (t / Ct, k - 1) / pow (gsl_sf_gamma (k), 2) / pow (m, 2*k)
    * exp (- L / m / sqrt(Ct));
}

static void
subprocess_func (int threadnum, int threadcount, void *user_data)
{
  SubprocessData *data = (SubprocessData *) user_data;
  RioData *lines = data->lines;
  gchar *classification = data->classification;
  float *likelihood = data->likelihood;
  int N = rio_data_get_num_entries (lines);

  double threshold = data->threshold;

  for (int i = threadnum * (N / threadcount);
       i < (threadnum + 1) * (N / threadcount);
       i++) {

    double p1;
    RioLine *l = rio_data_get_line (lines, i);
    switch (data->mode) {
    case MODE_AREA:
      p1 = prior_building_area (l,
                                data->param_lambda_x,
                                data->param_lambda_y,
                                data->param_k,
                                data->param_m);
      break;
    case MODE_AREA_LENGTH:
      p1 = prior_building_area_length (l,
                                       data->param_lambda_x,
                                       data->param_lambda_y,
                                       data->param_k,
                                       data->param_m);
      break;
    default:
      g_assert_not_reached ();
    }

    /* Classify */
    classification[i] = p1 > threshold;
    likelihood[i] = rio_htonf (p1);
  }
}

int
main (int argc, char **argv)
{
  int c, N, status;
  const char *infilename = NULL;
  const char *outfilename = NULL;
  RioData *lines = NULL;
  SubprocessData data;
  data.mode = MODE_AREA;
  data.threshold = DEFAULT_THRESHOLD;
  data.param_k = DEFAULT_PARAM_K;
  data.param_m = DEFAULT_PARAM_M;
  data.param_lambda_x = DEFAULT_PARAM_LAMBDA_X;
  data.param_lambda_y = DEFAULT_PARAM_LAMBDA_Y;

  while ((c = getopt (argc, argv, GETOPT_OPTIONS)) != -1) {
    switch (c) {
    case 'm':
      if (strcmp(optarg, "A") == 0) {
        data.mode = MODE_AREA;
      } else if (strcmp(optarg, "AL") == 0) {
        data.mode = MODE_AREA_LENGTH;
      } else {
        fprintf (stderr, "ERROR: Bad argument '%s' to -m option.\n\n",
                 optarg);
        usage (argv[0], 1);
      }
      break;
    case 't':
      status = sscanf (optarg, "%lf", &data.threshold);
      if (status != 1) {
        fprintf (stderr, "ERROR: Bad argument '%s' to -t option.\n\n",
                 optarg);
        usage (argv[0], 1);
      }
      break;
    case 's':
      status = sscanf (optarg, "%lf:%lf",
                       &data.param_lambda_x, &data.param_lambda_y);
      if (status != 2) {
        fprintf (stderr, "ERROR: Bad argument '%s' to -s option.\n\n",
                 optarg);
        usage (argv[0], 1);
      }
      break;
    case 'p':
      status = sscanf (optarg, "%lf:%lf",
                       &data.param_k, &data.param_m);
      if (status != 2) {
        fprintf (stderr, "ERROR: Bad argument '%s' to -p option.\n\n",
                 optarg);
        usage (argv[0], 1);
      }
      break;
    case 'j':
      status = sscanf (optarg, "%i", &rut_multiproc_threads);
      if (status != 1) {
        fprintf (stderr, "ERROR: Bad argument '%s' to -j option.\n\n",
                 optarg);
        usage (argv[0], 1);
      }
      break;
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
      g_assert_not_reached ();
    }
  }

  /* Warn if input doesn't make sense */
  if (data.mode == MODE_AREA_LENGTH
      && data.param_lambda_x != data.param_lambda_y) {
    fprintf (stderr,
"WARNING: In AL (area/length) mode, results may be unreliable without\n"
"equal ground range and azimuth scale factors.\n\n");
  }

  /* Get input filename */
  if (argc - optind < 1) {
    fprintf (stderr, "ERROR: You must specify an input filename.\n\n");
    usage (argv[0], 1);
  }
  infilename = argv[optind++];

  /* Get output filename, if specified */
  if (argc - optind > 0) {
    outfilename = argv[optind++];
  } else {
    outfilename = infilename;
  }

  /* Load ridge data */
  lines = rio_data_from_file (infilename);
  if (lines == NULL) {
    fprintf (stderr, "ERROR: Could not read ridge data from '%s': %s.\n",
             infilename, strerror (errno));
    exit (2);
  }
  if (rio_data_get_type (lines) != RIO_DATA_LINES) {
    fprintf (stderr, "ERROR: '%s' does not contain ridge line data.\n",
             infilename);
    exit (2);
  }
  N = rio_data_get_num_entries (lines);

  /* Create classification arrays. These are allocated using RutSurfaces
   * so that they can be modified by subprocesses. */
  data.classification = rut_multiproc_malloc (N * sizeof (gchar));
  data.likelihood = rut_multiproc_malloc (N * sizeof (float));
  data.lines = lines;

  /* Carry out multiprocess classification */
  if (!rut_multiproc_task (subprocess_func, &data)) {
    fprintf (stderr, "ERROR: Line data processing failed.\n");
    exit (3);
  }

  /* Save data */
  rio_data_set_metadata (lines, RIO_KEY_IMAGE_CLASSIFICATION,
                         data.classification, N * sizeof (gchar));
  rio_data_set_metadata (lines, RIO_KEY_IMAGE_CLASS_LIKELIHOOD,
                         (char *) data.likelihood, N * sizeof (float));
  if (!rio_data_to_file (lines, outfilename)) {
    fprintf (stderr, "ERROR: Could not write ridge data to '%s': %s.\n",
             outfilename, strerror (errno));
    exit (4);
  }

  rio_data_destroy (lines);
  rut_multiproc_free (data.classification);
  rut_multiproc_free (data.likelihood);
  return 0;
}
