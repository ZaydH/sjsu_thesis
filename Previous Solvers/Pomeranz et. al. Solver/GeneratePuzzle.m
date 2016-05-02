%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%                          The GeneratePuzzle  function takes an image and
%                          randomly scramble it
%
% Parameters:
%   src : The full path of the image to scramble
%   dest : The full path of where the scrambled image will be created
%   partSize : The part's size which will be used to cut the image into
%                            parts. Default value is 28.
%
%   Notice that the imwrite function might modify the image (even
%   when used with mode = lossless) and thus might affect the results.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Version : 1.0
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Programmer : Dolev Pomeranz
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function GeneratePuzzle(src, dest, partSize)

    % Input validation
    if (nargin == 0)
        error('GeneratePuzzle error - enter image source and image destination full paths');
    elseif (nargin == 1)
        error('GeneratePuzzle error - enter image destination full path as well');
    elseif (nargin == 2)
        partSize = 28;
    end

    image = imread(src);
    imageSize = size(image);
    fileExtension = lower(src(length(src)-2:length(src)));
    
    % Calculate a parts data
    rows = imageSize(1) / partSize;
    cols = imageSize(2) / partSize;
    numOfParts = rows * cols;
    
    rgbPartsArray =  zeros([partSize, partSize, 3, numOfParts], class(image));
    
     % Splits the image into parts.
    cutImageToParts();
    
    % Create a random order of parts
    partsOrder = randperm(numOfParts);
    
    outputImage = buildOutputImage(partsOrder);
    
    % Writing the image. Notice that it is important to add the lossless
    % mode, otherwise the image might change dramatically with respect to
    % the solver's performance. Using 'Quality' = 100 is not the same.
    imwrite(outputImage, dest, 'Mode', 'lossless');

                                                                            %%%%%% End of generate puzzle code %%%%%%
    
    % %%%%%%%%%%%%%
    % NESTED FUNCTIONS
    % %%%%%%%%%%%%%

    % Cuts the images into parts.
    function cutImageToParts()
        for index = 1 : numOfParts
            rowStartIndex = (ceil(index / cols)  - 1) * partSize + 1;
            rowEndIndex = rowStartIndex + (partSize -  1);
            colStartIndex = mod(index - 1, cols)  * partSize + 1;
            colEndIndex = colStartIndex + (partSize -  1);
            rgbPartsArray(:,:,:, index) = image(rowStartIndex :rowEndIndex, colStartIndex :colEndIndex, :);
        end
    end
    
    % Builds the ouptut image according to the parts order
    function [reconstructedImage] = buildOutputImage(order)
        reconstructedImage = zeros(imageSize, class(image));
        
         for i = 1 : numOfParts
             rowStartIndex = (ceil(i / cols)  - 1) * partSize + 1;
             rowEndIndex = rowStartIndex + (partSize -  1);
             colStartIndex = mod(i - 1, cols)  * partSize + 1;
             colEndIndex = colStartIndex + (partSize -  1);
            
             % If in order(i) we did not place a part yet, it is colored in
             % a special color in all pixels, to mark that a part was not
             % placed there.
             if (order(i) == 0)
                 reconstructedImage(rowStartIndex :rowEndIndex, colStartIndex :colEndIndex, :) = 0;
             else
                reconstructedImage(rowStartIndex :rowEndIndex, colStartIndex :colEndIndex, :) = ...
                        rgbPartsArray(:,:,:, order(i));
             end
         end
    end
end