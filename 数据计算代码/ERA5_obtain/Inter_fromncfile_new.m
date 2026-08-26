clc;clear all;
dirt='../era5_download/';
savepath='./';
monday=load('month.txt');
subdir=dir(dirt);
for i=1:length(subdir)
if(isequal(subdir(i).name,'.') || isequal(subdir(i).name,'..') )   
   continue;
end
fn=[dirt,subdir(i).name];
m=subdir(i).name;
year=str2num(m(6:9));
month=str2num(m(10:11));
day=str2num(m(12:13));
% if mod(year,4)==0
%     daofmonth=monday(:,2);
%     endday=daofmonth(month);
% else
%     daofmonth=monday(:,1);
%     endday=daofmonth(month);
% end
ncdisp(fn);
lon = ncread(fn, 'longitude');  nlon = length(lon);
lat = ncread(fn, 'latitude');   nlat = length(lat);
pre = ncread(fn, 'level');      npre = length(pre) ;
tim = single( ncread(fn, 'time')) ;       nepo = length(tim); 
[fip,fo] = meshgrid(lat,lon);
fi =double(fip)*pi/180;
g = 9.8062.*(1-0.0026442.*cos(2*fi)+0.0000058.*(cos(2*fi)).*(cos(2*fi)));
tim(1)
t0=datetime(1900,1,1);
rmjd = tim./24 + juliandate(t0).*ones(length(tim),1);

P = zeros(nlon, nlat, npre);
[P1, isort] = sort(pre, 'descend') ;
for ilon = 1 : nlon
    for ilat = 1 : nlat
        P(ilon, ilat, 1:npre) = double( P1 ) ;
    end
end

nout_layer = 4;
cfmt = '%10.4f %10.4f  ';
for i = 1 : nout_layer;    cfmt = [cfmt, '%6.1f ']; end  % h
for i = 1 : nout_layer;    cfmt = [cfmt, '%6.3f ']; end  % Lh
for i = 1 : nout_layer;    cfmt = [cfmt, '%6.3f ']; end % Lw
for i = 1 : nout_layer;    cfmt = [cfmt, '%8.1f ']; end  % P
for i = 1 : nout_layer;    cfmt = [cfmt, '%6.1f ']; end  % T
for i = 1 : nout_layer;    cfmt = [cfmt, '%6.1f ']; end % Tm
for i = 1 : nout_layer;    cfmt = [cfmt, '%10.3e ']; end % Q
% t1=datetime(year, month, day);
% t2=datetime(year, month,endday);

for iepo = 1:length(tim)
    %iepo = ( mjd - rmjd ) / 0.25 + 1
    z = ncread(fn, 'z', [1, 1, 1, iepo], [nlon, nlat, npre, 1 ] ); 
    %hgeop = z ./ 9.80665;
    hgeop = z ./ g;
    t = ncread(fn, 't',  [1, 1, 1, iepo], [nlon, nlat, npre, 1 ] ); 
    q = ncread(fn, 'q',  [1, 1, 1, iepo], [nlon, nlat, npre, 1 ] ); 
    hgeopi = hgeop(:,:,isort);
    ti =  t(:,:,isort);
    qi = q(:,:,isort);
    [Lh, Lw, Tm] = SHAtrop_layer(npre, hgeopi, P, ti, qi) ;    
    %[iyr, imon, iday, ihr, imin, isec] = mjd2utc( juliandate(t0)+tim(iepo)/24  ) ;
    [cyr, cmon, cday, chr, cmin, csec] = cUTC(year, month, day, iepo-1, 0, 0) ;
	fn_zpd = [savepath,cyr, cmon, cday, '.', chr, 'zpd'] ;
    fid2 = fopen(fn_zpd, 'wt');
    for i = 1 : size(Lh,1)
        for j = 1 : size(Lh, 2)
            fprintf(fid2, [cfmt, '\n'], lon(i), lat(j), ...
                hgeopi(i,j,1:nout_layer), ...
                Lh(i,j,1:nout_layer), Lw(i,j,1:nout_layer), ...
                P(i,j,1:nout_layer),  t(i,j,1:nout_layer), ...
                Tm(i,j,1:nout_layer), q(i,j,1:nout_layer) );
        end
    end
    
    fclose(fid2);
    clear z hgeop t q hgeopi ti qi Lh Lw Tm 
end
end