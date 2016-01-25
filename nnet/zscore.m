function [ output_args ] = zscore( images )
%ZSCORE Summary of this function goes here
%   Detailed explanation goes here
    output_args = bsxfun(@rdivide, bsxfun(@minus, images, mean(images)), std(images));
end

