All files as csv
"bA" = before annealing
"aA" = after annealing
"1500rpm" is the sample spin coated using 1500 rpm
"dcS1" is drop cast sample 1
"dcS2" is drop cast sample 2

Note: no "bAdcS1" file, since we do not have any spectral data for the dcS1 sample before annealing.



data file column name : explanation

t (s): time in seconds
S : sample transmitted signal after bleaching (changes over time)
dark : sample transmitted signal before bleaching (constant)
blank : uncoated sapphire transmitted signal
T_S : S divided by blank
T_dark : dark divided by blank
l (cm): approximate sample film thickness in cm
c_tot (M) : approximate sample total concentration of dye in M, as calculated using T_dark, epsilon and l