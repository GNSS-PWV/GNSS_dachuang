function [cyr, cmon, cday, chr, cmin, csec] = cUTC(iyr, imon, iday, ihr, imin, isec)
%UNTITLED6 此处显示有关此函数的摘要
%   此处显示详细说明
cyr = num2str(iyr);
if imon >= 10
   cmon = num2str(imon);
else
   cmon = ['0', num2str(imon)]
end
if iday<10
   cday=['0',num2str(iday)];
else
   cday=[num2str(iday)];
end
if ihr<10
   chr=['0',num2str(ihr)];
else
   chr=[num2str(ihr)];
end
if imin<10
   cmin=['0',num2str(imin)];
else
   cmin=[num2str(imin)];
end
if isec<10
   csec=['0',num2str(isec)];
else
   csec=[num2str(isec)];
end
end

