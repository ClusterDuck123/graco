#include <iostream>
#include <fstream>
#include <numeric>
#include <cmath>
#include <array>
#include <fstream>
#include <cstdlib>
#include <algorithm>

// Global variables
using std::fstream;
fstream fin, fout;
int p;

/*
==================================
  Functions and global variables
==================================
*/

// Functions
double* get_weights();
void write_results(double** D, int m);

/*
=================
  init function
=================
*/
int init(int argc, char const *argv[]) {
// open input and output files
	if (argc!=4) {
		std::cerr << "Incorrect number of arguments." << '\n';
		std::cerr << "Usage: tijana [p] [input file] "
                 "[output file]" << '\n';
		return 0;
	}

  p = atoi(argv[1]);
	fin.open(argv[2], fstream::in);
	fout.open(argv[3], fstream::out | fstream::binary);
	if (fin.fail()) {
		std::cerr << "Failed to open file " << argv[2] << '\n';
		return 0;
	}
	if (fout.fail()) {
		std::cerr << "Failed to open file " << argv[3] << '\n';
		return 0;
	}
  return 1;
}


/*
=================================
              main
=================================
*/
int main(int argc, char const *argv[]) {

// import file
  if (!init(argc, argv)) {
		std::cerr << "Stopping!" << '\n';
		return 0;
	}

// read matrix in
  char c;
  int m, n;
  fin >> c >> m >> n;
// assert c == '#'
// assert n == 15
  long long int GDV[m][n];
  for (auto i = 0; i < m; i++) {
    for (auto j = 0; j < n; j++) {
      fin >> GDV[i][j];
    }
  }

  /*
  =============================
    Calculate distance matrix
  =============================
  */

  // Create empty distance matrix
  double **D = new double*[m];
  for(auto i = 0; i < m; ++i)
    D[i] = new double[m];

	if (p == 0) {
		double sum = 0;
	  for (auto u = 0; u < m; u++) {
	    for (auto v = u+1; v < m; v++) {
	      double max = 0;
				long double denom;
				unsigned long long int nom;
	      for (auto i = 0; i < n; i++) {
					denom = std::abs(GDV[u][i] - GDV[v][i]);
					nom   = std::abs(GDV[u][i]) + std::abs(GDV[v][i]);
					if (nom != 0) max = std::max(double(denom)/nom, max);
	      }
	      D[u][v] = max;
	      D[v][u] = D[u][v];
			}
		}
	}

	else {
	  double sum = 0;
	  for (auto u = 0; u < m; u++) {
	    for (auto v = u+1; v < m; v++) {
	      sum = 0;
	      for (auto i = 0; i < n; i++) {
					double denom = std::abs(GDV[u][i] - GDV[v][i]);
					double nom   = std::abs(GDV[u][i]) + std::abs(GDV[v][i]);
					if (nom != 0) sum += std::pow(denom/nom, p);

	      }
	      D[u][v] = std::pow(sum, 1./p);
	      D[v][u] = D[u][v];
	    }
	  }
	}


  write_results(D, m);

  return 0;
}

/*
=================
  write_results
=================
*/
void write_results(double** D, int m){
  for (auto i=0;i<m;i++) {
		for (auto j=0;j<m;j++)
			fout << D[i][j] << ' ';
    fout << '\n';
	}
	fout.close();
}
