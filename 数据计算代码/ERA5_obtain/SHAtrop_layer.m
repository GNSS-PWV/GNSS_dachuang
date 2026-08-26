function [Lh, Lw, Tm] = SHAtrop_layer(nlayer, H, P, T, Q)

k1 = 77.689;  % K/hPa
k2 = 71.2952;  % K/hPa
k3 = 375463;  % K^2/hPa
% % % % % % % % Bevis, 1994
% k1 = 77.6;  % K/hPa
% k2 = 70.4;  % K/hPa
% k3 = 373900;  % K^2/hPa

% % % universal gas constant R, molecular weight of water Mw and dry air Md
% % % see Ref. 1, pp 8
R = 8.3143 ; % J/K/mod
Md = 28.965 ; % g/mol
Mw = 18.016; % g/mol


% p, air density, hPa, see Ref. 2
Rd = 287.053 ; % J K^-1 kg^-1 gas constant for dry air, should = R/Md*1000
Rw = 461.495 ; % J K^-1 kg^-1 gas constant for water vapor,  should = R/Mw*1000


dtmp = Rd/Rw;
e = Q .* P ./ ( dtmp + (1-dtmp) .* Q ) ;


% dry atmosphere density, hPa
Pd = P - e;
 
% air density for sum of dry and wet atmosphere
p = Pd ./ T ./ Rd  + e ./ T ./ Rw  ;


Nh = k1 .* Rd .* p ;
Nw = ( k2 - k1*Rd/Rw ) .* e ./ T + k3 .* e ./ T.^2 ;


% ilon = 50; ilat = 60;
% dx = reshape( H(ilon, ilat, :), 1, 37 ) ;
% dy = reshape( Nh(ilon, ilat, :), 1, 37 ) ;
% plot(dx,dy)
% return



for il  = 1 : 2
    if il == 1; Ni = Nh; end
    if il == 2; Ni = Nw; end
    B = diff( log(Ni), 1, 3 ) ./ diff( H, 1, 3 );
    A = Ni(:, :, 1:end-1) ./ exp( B.*H(:, :, 1:end-1) ) ; %Nh(2:end) ./ exp( Bh.*H1(2:end) ) ;
    L =  A ./ B .* ( exp(B.*H(:, :, 2:end)) - exp(B.*H(:, :, 1:end-1)) )  ;
%     tic
    for i = nlayer-2 : -1 : 1
        L(:,:,i) = L(:,:,i+1) + L(:,:,i) ;
    end
%     toc
    if il == 1; Lh = L  * 10^-6; end
    if il == 2; Lw = L  * 10^-6; end
    clear B A L Ni
end


Tm1 = e ./ T ;
Tm2 = e ./ T.^2 ;
for i = nlayer-2 : -1 : 1
    Tm1(:, :, i) = Tm1(:,:,i+1) + Tm1(:,:,i) ;
    Tm2(:, :, i) = Tm2(:,:,i+1) + Tm2(:,:,i) ;
end
Tm = Tm1 ./ Tm2;



return

ilon = 50; ilat = 60;
dx = reshape( H(ilon, ilat, :), 1, 37 ) 
dy = reshape( Lh(ilon, ilat, :), 1, 36 ) ;
plot(dx(1:36),dy)

end

