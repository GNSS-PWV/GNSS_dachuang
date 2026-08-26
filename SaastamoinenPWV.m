function PWV = SaastamoinenPWV(lat, h, P, e, T, Tm)
%------------------------------------------------------------
% Saastamoinen模型计算ZTD、ZHD、ZWD及PWV
%
% 输入：
%   P    地面气压(hPa)
%   T    地面温度(K)
%   e    水汽压(hPa)
%   lat  纬度(°)
%   h    高程(m)
%   Tm   大气加权平均温度(K)
%
% 输出：
%   PWV  大气可降水量(mm)
%------------------------------------------------------------

%% 常数
k2p = 22.1;          % 大气折射常数，K/hPa
k3 = 3.739e5;       % 大气折射常数，K^2/hPa
Rv = 461.495;         % 水汽气体常数，J
rho_w = 1000;       % 液态水密度，kg/m^3

%% 单位转换
lat = deg2rad(lat);  %纬度lat角度转弧度
T = T + 273.15;      %气温T摄氏度转开尔文温度

%% 重力修正项f
% h输入单位为m，因此需转换为km
f = 1 - 0.00266*cos(2*lat) - 0.00000028*h;

%% Saastamoinen模型计算天顶总延迟ZTD(m)
ZTD = 0.002277 * ( P/f + (0.05 + 1255/T) * e ) * 1000;

%% Saastamoinen模型计算天顶干延迟ZHD(m)
ZHD = 0.002277 * P / f * 1000;

%% 相减得天顶湿延迟ZWD(m)
ZWD = ZTD - ZHD;

%% 转换因子Π
PI = 1e8 / (rho_w * Rv * (k3/Tm + k2p));

%% PWV
% ZWD单位为m，PWV输出为mm
PWV = PI * ZWD;

end