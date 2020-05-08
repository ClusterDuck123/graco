#include <iostream>
#include <fstream>
#include <numeric>
#include <cmath>
#include <array>
#include <fstream>
#include <cstdlib>
#include <algorithm>

// Global variables
const double _LOG2 = 0.69314718055;

using std::fstream;
fstream fin, fout;


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
	if (argc!=3) {
		std::cerr << "Incorrect number of arguments." << '\n';
		std::cerr << "Usage: hellinger [input file] "
                 "[output file]" << '\n';
		return 0;
	}

	fin.open(argv[1], fstream::in);
	fout.open(argv[2], fstream::out | fstream::binary);
	if (fin.fail()) {
		std::cerr << "Failed to open file " << argv[1] << '\n';
		return 0;
	}
	if (fout.fail()) {
		std::cerr << "Failed to open file " << argv[2] << '\n';
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
  double GDV[m][n];
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



  double sum = 0;
  for (auto u = 0; u < m; u++) {
    for (auto v = u+1; v < m; v++) {
      sum = 0;
      for (auto i = 0; i < n; i++) {
				double p_i = GDV[u][i];
				double q_i = GDV[v][i];
				double m_i = (p_i+q_i)/2;

				double p_log = 0;
				double q_log = 0;
				if (p_i != 0) {p_log = std::log(p_i/m_i);}
				if (q_i != 0) {q_log = std::log(q_i/m_i);}

				sum += (p_i*p_log + q_i*q_log)/2;
      }

      D[u][v] = sum / _LOG2;
      D[v][u] = D[u][v];
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
