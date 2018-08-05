function n=zero_norm(x, threshold)
    n=0;
    for i=1:max(size(x, 1), size(x, 2))
        if abs(x(i))>threshold
            n=n+1;
        end
    end
end