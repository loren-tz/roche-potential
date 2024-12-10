#include <stdio.h>
#include <stdlib.h>
#define _USE_MATH_DEFINES
#include <math.h>

#define GAMMA (4 * (M_PI * M_PI))

struct vector {
  double x;
  double y;
};

double cub_mod(struct vector);
struct vector scal(struct vector, double);
struct vector sum(struct vector, struct vector);
struct vector diff(struct vector, struct vector);
double mod(struct vector);

int main (int argc, char *argv[]) {
  FILE *fp;
  struct vector r_a, r_b, r_ab, point;
  double mu_a, mu_b, Roche_pot, x, y, q;
  double M1 = atof(argv[1]);
  double M2 = atof(argv[2]);
  double L = atof(argv[3]);
  double R = atof(argv[4]);
  double cutoff = atof(argv[5]);

  if(M1 <= M2) {
    q =  M1 / (M1 + M2);
  } else {
    q = M2 / (M1 + M2);
  };
  mu_a = q;
  mu_b = 1 - mu_a;
  fp = fopen("roche_data.dat", "w");
  
  r_a.x = -mu_a * cos(M_PI * 0.25);
  r_a.y = -mu_a * sin(M_PI * 0.25);
  r_b.x = mu_b * cos(M_PI * 0.25);
  r_b.y = mu_b * sin(M_PI * 0.25);

  r_ab = diff(r_a, r_b);

  L *= 0.5;
  
  for(x = -L; x < L; x += R) {
    for(y = -L; y < L; y += R) {
        point.x = x;
        point.y = y;
        Roche_pot = -(((GAMMA * mu_b) / mod(diff(r_a, point))) + ((GAMMA * mu_a) / mod(diff(r_b, point))) + (0.5 * ((GAMMA / cub_mod(r_ab)) * (mod(point) * mod(point)))));
      if(Roche_pot >= cutoff) {
        fprintf(fp, "%lf %lf %lf\n", x, y, Roche_pot);
      }
    }
  }

  fclose(fp);
}

double cub_mod(struct vector r) {
  double mod_r = sqrt((r.x * r.x) + (r.y * r.y)) * sqrt((r.x * r.x) + (r.y * r.y)) * sqrt((r.x * r.x) + (r.y * r.y));
  return mod_r;
}

struct vector scal(struct vector n, double s) {
  n.x = n.x * s;
  n.y = n.y * s;
  return n;
}

struct vector sum(struct vector n, struct vector s) {
  n.x = n.x + s.x;
  n.y = n.y + s.y;
  return n;
}

struct vector diff(struct vector n, struct vector s) {
  n.x = n.x - s.x;
  n.y = n.y - s.y;
  return n;
}

double mod(struct vector r) {
  double mod_r = sqrt((r.x * r.x) + (r.y * r.y));
  return mod_r;
}
