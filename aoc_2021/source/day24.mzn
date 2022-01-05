include "globals.mzn"; 

int: n = 14; % number of digits
array[1..n] of int: xadd = [13, 11, 14, -5, 14, 10, 12, -14, -8, 13, 0, -5, -9, -1];
array[1..n] of int: yadd = [0, 3, 8, 5, 13, 9, 6, 1, 1, 2, 7, 5, 8, 15];
array[1..n] of int: zdiv = [1, 1, 1, 26, 1, 1, 1, 26, 26, 1, 26, 26, 26, 26];
%int: m = 2; % max number of steps to take

% decision variables
array[1..n] of var 1..9: digits;
array[1..n+1] of var 0..1000000000: zs;
array[1..n] of var 0..1000000000: zprevs;

constraint
  zs[15] = 0
  /\
  forall(i in 1..n)(
    zs[i] >= 0
    )
    /\
    zprevs[1] = 0
    /\
    forall(i in 1..n)(
      if zdiv[i] = 1 
      then zprevs[i] = zs[i-1 +1 ]
      else zprevs[i] = (zs[i-1 +1] div 26)
      endif
      )
    /\
    forall(i in 1..n)(
      if ((((zs[i-1+1] mod 26) + xadd[i]) = digits[i]) = 0 )
      then
          zs[i+1] = 26 * zprevs[i] + digits[i] + yadd[i]
      else
          zs[i+1] = zprevs[i]

      endif 
      )
      

;
solve minimize(sum(digits));

%solve :: int_search(digits, first_fail, indomain_min, complete) satisfy;

output [show(digits), show(zs)];