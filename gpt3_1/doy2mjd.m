function mjd = doy2mjd(year, doy)
    % year+doy -> mjd
    arguments
        year (1,1) double
        doy (1,1) double
    end

    % 构造datetime对象
    dt = datetime(year, 1, 1, 'Format', 'yyyy-MM-dd') + days(doy - 1);
    
    % 转换为mjd
    mjd = juliandate(dt) - 2400000.5;
end