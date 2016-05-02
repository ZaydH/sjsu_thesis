%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%                          The JigsawSolver function solves square jigsaw puzzles.
%
% Parameters:
% ==========
%   I :  The image full path
%   firstPartToPlace : The index of the first part to place as the seed
%       of the solution. Enter a number from 1 to the number of parts, or 0
%       to allow the solver to choose, or -1 for a random seed.
%   partSize : A square part size, this size should divide with the image
%        size or the image is cropped.
%   compatibilityFunc : The type of compatibility function to use
%                                                    0 - Dissimilarity
%                                                    1 - Prediction-based Compatibility (Default)
%                                                    [p, q] - Lpq norm. Where p and q are integers
%   shiftMode : Which shift module to use
%                               0 - None
%                               1 - Move largest segment across all possible locations and take the best placement
%                               2 - Growing largest segment using estimationFunc  estimation metric to 
%                                    measure convergence (Default)
%                               3 - Growing largest segment but with fixed 
%                               number of iterations
%   colorScheme : In which color scheme processing will be done
%                                   0 - RGB
%                                   1 - LAB (Default)
%                                   2 - HSV
%   greedyType :  The greedy solver criterion for choosing which part to
%                                   place next on the board
%                                   0 - Multiplication  compatibility
%                                   1 - Average compatibility
%                                   2 - Narrow Best Buddies with multiplication
%                                   3 - Narrow Best Buddies with average (Default)
%   estimationFunc : Which type of estimation metric will be used for conversion
%                                   0 - Best Buddies (Default)
%                                   1 - Average of compatibilities
%   outputMode : The user desired output 
%                                   0 - None (only return values)
%                                   1 - Output image only (Default)
%                                   2 - Output image + save figures and data to resPath
%                                   3 - Create a visualization movie in
%                                        resPath named 'JigsawSolverVisualization.avi' (will not activate solver!!)
%   runTests : Should run in test mode (boolean value). Test mode runs
%                            basic unit tests and compatibility function tests. The default is false.
%   normalMode :  What method should be used for compatibility normalization
%                                   1 - Linear according to the max value
%                                   2 - Same as used by Cho et al
%                                   3 - Exponent according to median value
%                                   4 - Exponent according to first quartile value (Default)
%   resPath : The path where resources will be created. 'C:\temp\' is the default
%   debug : debug arguments (Enables you to see each of the solver's placements visually)
%                       0 - Off (Default)  
%                       1 - Manual (the user can control the solver's progress)
%                       2 - Manual + create debug movie in resPath named 'JigsawSolverInAction.avi'
%                       3 - Automatic (the user can only see the progress)
%                       4 - Automatic + create debug movie in resPath named 'JigsawSolverInAction.avi'
%
% Return values:
% ============
%   directCompVal : The solution's direct comparison value
%   neighborCompVal : The solution's neighbor comparison value
%   estimatedBestBuddiesCorrectness : Best buddies correctness estimation metric value
%   estimatedAverageOfProbCorrectness : Correctness estimation via average metric value
%   solutionPartsOrder : The solver's solution order (left -> right, top ->   bottom order)
%   segmentsAccuracy : What is the percentage of correct parts in the largest segment
%   numOfIterations : The number of iteration it took the solver to converge
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Version : 2.19
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Programmer : Dolev Pomeranz
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [directCompVal, neighborCompVal, estimatedBestBuddiesCorrectness, estimatedAverageOfProbCorrectness, solutionPartsOrder, segmentsAccuracy, numOfIterations] = ...
        JigsawSolver(I, firstPartToPlace, partSize, compatibilityFunc, shiftMode, colorScheme, greedyType, estimationFunc, outputMode, runTests, normalMode,resPath, debug)
    
    %tic;
    
    % This enables the puzzle to float until it reaches the proper size. 
    % Allowing more accurate decisions and to solve the shifiting problem.
    floatingPuzzleMode = 1;

    % Checks if we should run in tests mode.
    if (nargin > 9) && (runTests) 
        setOutputToSolverNotActivated();
        
        testsPassed = 0;
        mainCodeTester();
        %toc;
        return
    end
    
	% initalize default values (in case the user ommited parameters)    
	initDefaultValues(nargin);
    
    % Reading the image and crop it to be in a valid size.
    image = imread(I);
    imageSize = size(image);
    imageSize(1) = imageSize(1) - mod(imageSize(1), partSize);
    imageSize(2) = imageSize(2) - mod(imageSize(2), partSize);
    image = imcrop(image,[1, 1, imageSize(2) - 1, imageSize(1) - 1]);
    
    imageWithNewColorScheme = convertToUserSelectedColorScheme();
    
    image = double(image);
    imageSize = size(image);
    outputImage = zeros(imageSize);
    onlyGreedyAlgoImage = zeros(imageSize);
    figNum = 1;
    numOfIterations = 1;
    onlyGreedyDirectCompVal = -1;
    onlyGreedyNeighborCompVal = -1;
    
    % Calculate a parts data
    rows = imageSize(1) / partSize;
    cols = imageSize(2) / partSize;
    numOfParts = rows * cols;
    partsCorrectOrder = 1:numOfParts;
    
    % Creating parts array that contain the puzzle parts
    newColorSchemePartsArray =  zeros(partSize, partSize, 3, numOfParts);
    rgbPartsArray =  zeros(partSize, partSize, 3, numOfParts);
    
    % Splits the image into parts.
    cutImageToParts();
    
    % Algorithm data structures
    partsCompVal = zeros(numOfParts,numOfParts,4);
    confidence = zeros(numOfParts, numOfParts, 4) ;
    partsMat = zeros(rows,cols);
    neighborVec = zeros(3, numOfParts);
    unplacedParts = 1:numOfParts; % Consider - randperm(numOfParts);
    unplacedCounter = numOfParts;
    segments = 0;
    segmentsAccuracy = [0,0];
    bestNeighbors = zeros(numOfParts, 4);
    bestBuddies = zeros(2, numOfParts);
    floatActiveInAxe = [1,1];
    [pathstr, name, ext] = fileparts(I);
    
    % In case the user only wants a visulaztion movie.
    if (outputMode == 3)
        setOutputToSolverNotActivated();
        createVisualizationMovie();
        return
    end
    
    % If the user is in debug mode all other output is disabled.
    if (debug)
        outputMode = 0;
    end
    
    % This debug mode will generate a debug movie.
    framePerSecond = 4;
    if (debug == 2) || (debug == 4)
        aviobj = avifile(strcat(resPath,'JigsawSolverInAction.avi'),'fps',framePerSecond,'compression', 'None');
    end
    
    % The prediction-based compatibility is marked by two flags: '22' because
    % of "historical" reasons and '1' a user friendly value.
    if (compatibilityFunc == 1)
        compatibilityFunc = 22;
    end
    
    % Calculate the parts compatibility.
    initializePartsCompatibility();
    initializeConfidence();
    
    % As oppose to the threashold method.
    initBestNeighbors();
    
    if (nargin <= 1) || (firstPartToPlace == -1)
        firstPartToPlace = ceil(rand * numOfParts);
    elseif (firstPartToPlace == 0)
        % Estimate the best part to start as the greedy seed.
        firstPartToPlace = selectBestSeedCandidate();
    end
    
    % Define the name for this execution
    imageName = strcat('Image-', name, ext, ...
                                        '_Parts-', num2str(numOfParts), ... 
                                        '_Size-', num2str(partSize), '_Seed-', num2str(firstPartToPlace), ...
                                        '_Comp-', num2str(compatibilityFunc), '_Shift-', num2str(shiftMode), ...
                                        '_Color-', num2str(colorScheme), '_GreedyType-', num2str(greedyType),'_EstimationFunc-', num2str(estimationFunc), '_NormalMode-',num2str(normalMode));
    
    % Optional - use relaxation labeling to improve the confidence level
    % else we will use the "zero" iteration of the RL process.
    
    % Activate the greedy algorithm for placing the parts (This will create
    % the parts placing matrix).
    greedyPlacingAlgorithm();    
    
    % Optional - use shifting algorithm to improve results (This will
    % update the parts placing matrix).
    shiftJigsawParts();
    
    % Create the  parts order from the parts placing matrix
    partsOrder = convertPartsMatToVec();
    
    % Compare the output image with actual image.
    [directCompVal, neighborCompVal] = jigsawComparison(partsOrder);
    estimatedBestBuddiesCorrectness = estimateCorrectnessByBestBuddy();
    estimatedAverageOfProbCorrectness = estimateCorrectnessAverageOfProb();
    solutionPartsOrder = partsOrder;
    
    % Build the output image and display the result
    if (outputMode)
        outputImage = buildOutputImage(partsOrder);
        
        % Save only if outputMode >  1. No caption needed if this is part
        % of a movie (debug ~= 2).
        displayImage(outputImage, (outputMode > 1), (debug ~= 2)); 
        
        if (outputMode == 2)
            % Saving the data for future use.
            save(strcat(resPath, imageName, ...
                                        '[D- ', num2str(directCompVal), '_N-', num2str(neighborCompVal), '].mat'), ...
                                        'outputImage', 'segments', 'onlyGreedyAlgoImage', 'partSize','firstPartToPlace', ...
                                        'compatibilityFunc', 'shiftMode', 'colorScheme', 'directCompVal', 'neighborCompVal', ...
                                        'onlyGreedyDirectCompVal','onlyGreedyNeighborCompVal', 'name', 'ext', 'numOfParts', ...
                                        'rows', 'cols', 'greedyType', 'estimationFunc', 'partsOrder', 'numOfIterations');
        end
    end
    
    if (debug == 2) || (debug == 4)
        % Add aditional frames of the last figure.
        frame = getframe(gca);
        for k = 1:10
            aviobj = addframe(aviobj,frame);
        end
        aviobj = close(aviobj);
    end
    
    %toc;
                                                                            %%%%%% End of solver code %%%%%%
    
    % %%%%%%%%%%%%%
    % NESTED FUNCTIONS
    % %%%%%%%%%%%%%
    
    % SHIFT CORRECTION VIA SEGMENTATION
    % %%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Initialize  the best neighbors data structures.
    % Noitce that we will use the confidence values and not the raw
    % parts compatibility values, since the confidence  values could 
    % have been improved, by RL for example.
    function initBestNeighbors()
        % Set the confidence diagonal as zero, inorder that it will not be
        % selected by the max function. It is irrelevent to the calculation.
        for i = 1:numOfParts
            for label = 1:4
                confidence(i, i, label) = 0;
            end
        end
        
        for current = 1:numOfParts
            for label = 1:4
                [dummy, maxIndexVec] = max(confidence(current, :, label));
                bestNeighbors(current, label) = maxIndexVec(1);
            end
        end
    end
    
    % Decides if the cell in location (i,j) has a part that belongs to
    % segment number segCounter. Segmentation predicat.
    % A cell is in the segment if the part in it is the best part to place
    % according to the neighbor parts (Best Buddy - best from both ways)
    function [result] = isInSegment(i, j, segCounter)
        result = 1;
        
        if (i > 1) && (segments(i-1,j) == segCounter)
            result = result && bestNeighbors(partsMat(i,j), 3) == partsMat(i-1,j) && ...
                                bestNeighbors(partsMat(i-1,j), 4) == partsMat(i,j);
        end
        if (i<rows) && (segments(i+1,j) == segCounter)
            result = result && bestNeighbors(partsMat(i,j), 4) == partsMat(i+1,j) && ...
                                 bestNeighbors(partsMat(i+1,j), 3) == partsMat(i,j);
        end
        if (j > 1) && (segments(i,j-1) == segCounter)
            result = result && bestNeighbors(partsMat(i,j), 1) == partsMat(i,j-1) && ...
                                 bestNeighbors(partsMat(i,j-1), 2) == partsMat(i,j);
        end
        if (j<cols) && (segments(i,j+1) == segCounter)
            result = result && bestNeighbors(partsMat(i,j), 2) == partsMat(i,j+1) && ...
                                 bestNeighbors(partsMat(i,j+1), 1) == partsMat(i,j);
        end
    end
    
    % Creates a segment map of regions in the puzzle that are most likely
    % to be placed correctly but might be shifted.
    function [segCounter] = Segment()
        % Initialize segments as zeros
        segments = zeros(rows, cols);
        segCounter = 1;

        while  (~isempty(find(segments == 0)))
            unkownCells =  find(segments == 0);
            index = unkownCells(1);

            % Initalize current segment, each cell in the queue marks a
            % cell in the segment.
            queue = index;
            segments(index) = segCounter;
        
            while (~isempty(queue))
                % Pop from the queue.
                current = queue(1);
                queue = queue(2:end);
                
               % Transfer from current index to (i,j)
               i = mod(current - 1, rows)  + 1;
               j = floor((current - 1) / rows) + 1;

                % Adding any neighbors that have a high probability that
                % they are the correct neighbors.
                for di = -1:1
                    for dj = -1:1
                        % This condition will leave us with the vaild neighbors.
                        if (di * dj == 0 && di ~= dj) && (i + di >= 1 && i + di <= rows && j + dj >= 1 && j + dj <= cols)
                            neighbor = (j - 1 + dj) * rows + (i + di);
                            
                            % We will only consider unmarked cells.
                            if (segments(neighbor) == 0) 
                                % Checks if this candidate is in the
                                % segment
                                if (isInSegment(i+di, j+dj, segCounter))
                                    segments(neighbor) = segCounter;
                                    queue = [queue, neighbor];
                                else
                                    % Mark with temp sign of not in current
                                    % segment.
                                    segments(neighbor) = -1;
                                end
                            end
                        end
                    end
                end
            end
            
            % Restore temp sign as not decided
            segments(segments == -1) = 0;
            
            segCounter  =  segCounter + 1;
        end
    end

    % Finds the frame size of a wanted segment, according to the segments
    % map.
    function [x1, y1, x2, y2] = findFrameSize(segmentNumber)
        for x1 = 1:rows
            if (any(segments(x1,:) == segmentNumber))
                break
            end
        end
        for x2 = rows:-1:1
            if (any(segments(x2,:) == segmentNumber))
                break
            end
        end
        for y1 = 1:cols
            if (any(segments(:,y1) == segmentNumber))
                break
            end
        end
        for y2 = cols:-1:1
            if (any(segments(:,y2) == segmentNumber))
                break
            end
        end
    end
    
    % The shifter main method, iterate until the we reach the best parts
    % map according to the estimation metrics.
    function [bestPartsMat] = convergeWithSegmentation()
        NUM_ATTEMPTS = 1;
        attempts = NUM_ATTEMPTS;
        
        % Initialize the best as the output of the greedy algo.
        bestPartsMat = partsMat;
        bestPartsMatScore = estimatePartsMatCorrectness();
        
        floatActiveInAxe = [1,1];
        [newPartsMat, newPartsMatScore] = shiftSegments(0,1);
        numOfIterations = numOfIterations + 1;
        
        if (shiftMode == 2)
            while (attempts > 0)
                attempts = attempts - 1;
                
                if (newPartsMatScore > bestPartsMatScore)
                    bestPartsMatScore = newPartsMatScore;
                    bestPartsMat = newPartsMat;
                    attempts = NUM_ATTEMPTS;
                end
            
                floatActiveInAxe = [1,1];
                [newPartsMat, newPartsMatScore] = shiftSegments(0,0);
                numOfIterations = numOfIterations + 1;
            end
        elseif (shiftMode == 3)
            save = 0;
            for j = 1:numOfParts
                bestPartsMat = newPartsMat;
                floatActiveInAxe = [1,1];
                
                if (j == numOfParts)
                    save = 1;
                end
                
                [newPartsMat] = shiftSegments(0, save);
            end
        end
    end

    % Calculates the accuracy of the biggest segment. It means the
    % precentage of correct negihbors in the segment.
    function [accuracy] = biggestSegmentAccuracy(biggestSegMat)
        [frameRows, frameCols] = size(biggestSegMat);
        hits = 0;
        trials = 0;
        
        for i = 1:frameRows - 1
            for j = 1:frameCols - 1
                if biggestSegMat(i,j) ~= 0 && biggestSegMat(i,j + 1) ~= 0
                    trials = trials + 1;
                    
                    if biggestSegMat(i,j) == biggestSegMat(i,j + 1) - 1
                        hits = hits + 1;
                    end
                end
                
                if biggestSegMat(i,j) ~= 0 && biggestSegMat(i+1,j) ~= 0
                    trials = trials + 1;

                    if biggestSegMat(i,j) == biggestSegMat(i+1,j) - cols;
                        hits = hits + 1;
                    end
                end
            end
        end
        
        accuracy = hits / trials;
    end

    % Shift correction via segmentation algorithm
    function [bestPartsMat, bestPartsMatScore] = shiftSegments(tryAllPossiblePlacements, saveSegMap)
        % Create the segments map
        segCounter = Segment();
        
        % If needed show and save the segment map figure
        if (saveSegMap) && (outputMode == 2) 
            % Permutate the segmentations' numbers. So when displayed they
            % will be shown clearly. The problem is that the segmentation
            % algorithm gives neighbor segments a similar segment value.
            segmentsPermutation = segments;
            permutation = randperm(segCounter);
            
            for i = 1:segCounter
                segmentsPermutation(segments == permutation(i)) = i;
            end
            
            % Showing and saving the segmentation map.
            h = imagesc(segmentsPermutation);
            axis image
            axis off
            saveas(h, strcat(resPath,imageName,'(', num2str(figNum), ').fig'));
            figNum = figNum + 1;
        end

        % Store segments size.
        frames = zeros(segCounter, 1);
        for i = 1:segCounter
            frames(i) = length(find(segments == i));
        end

        % Find biggest segment frame
        [partsInSegment, biggestSegmentVec] = max(frames);
        biggestSegment = biggestSegmentVec(1);
        biggestSegmentLocation = (segments == biggestSegment);
        biggestSegmentNdx = find(biggestSegmentLocation == 1);
        [x1, y1, x2, y2] = findFrameSize(biggestSegment);
        biggestSegMat = zeros(rows, cols);
        biggestSegMat(biggestSegmentLocation) = partsMat(biggestSegmentLocation);
        
        % Reduce the segment frame to it's minimum size.
        biggestSegMat = biggestSegMat(x1:x2, y1:y2);
        [frameRows, frameCols] = size(biggestSegMat);
        
        segmentsAccuracy(1) = segmentsAccuracy(1) + biggestSegmentAccuracy(biggestSegMat);
        segmentsAccuracy(2) = segmentsAccuracy(2) + 1;
        
        % Init the unplaced parts vec, the biggest segment will be marked
        % as placed. Save in a copy variable for future reuse.
        unplacedPartsCopy = zeros(numOfParts, 1);
        unplacedPartsCopy((numOfParts - partsInSegment + 1) : numOfParts) = ...
            partsMat(biggestSegmentLocation);
        unplacedPartsCopy(1:numOfParts - partsInSegment) = ...
            setdiff(1:numOfParts, partsMat(biggestSegmentLocation));
        
        % Move the frame, after each movment relocate the parts, store the
        % best arrangment data.
        bestPartsMatScore = 0;
        bestPartsMat = zeros(rows, cols);
        
        % This flag indicates if this function should find the best
        if (tryAllPossiblePlacements == 0)
            
            % Place the biggest segment in the location (1,1).
            partsMat = zeros(rows,cols);
            partsMat(1:frameRows, 1:frameCols) = biggestSegMat;
            
            % Set the unplaced data structure without the segments
            % parts.
            unplacedCounter = numOfParts - partsInSegment;
            unplacedParts = unplacedPartsCopy;
            
             % Set the neighbor vec with the segment's neighbors.
            neighborVec = zeros(3, numOfParts);
            for index = 1:partsInSegment
                % Transfer from current index to (i,j)
                x = mod(biggestSegmentNdx(index) - 1, rows)  - (x1 - 1) + 1;
                y = floor((biggestSegmentNdx(index) - 1) / rows) - (y1 - 1) + 1;
                addPartNeighbors(x,y);
            end

            % Place the remaining parts (Using a greedy algorithm)
            placeRemainingParts();
            
            bestPartsMatScore =  estimatePartsMatCorrectness();
            bestPartsMat = partsMat;
        else
            % NOTICE - At this point it is best if the floating mode is
            % inactive, since we want the largest segment to hold its
            % position. Thus, floatActiveInAxe == [0,0]
            
            for i = 1:(rows - frameRows + 1)
                for j = 1:(cols - frameCols + 1)
                    % Place the biggest segment in the current location (i,j).
                    partsMat = zeros(rows,cols);
                    partsMat(i:(i + frameRows - 1), j:(j + frameCols - 1)) = biggestSegMat;

                    % Set the unplaced data structure without the segments
                    % parts.
                    unplacedCounter = numOfParts - partsInSegment;
                    unplacedParts = unplacedPartsCopy;

                    % Set the neighbor vec with the segment's neighbors.
                    neighborVec = zeros(3, numOfParts);
                    for index = 1:partsInSegment
                        % Transfer from current index to (i,j)
                        x = mod(biggestSegmentNdx(index) - 1, rows)  - (x1 - 1) + i;
                        y = floor((biggestSegmentNdx(index) - 1) / rows) - (y1 - 1) + j;
                        addPartNeighbors(x,y);
                    end

                    % Place the remaining parts (Using a greedy algorithm)
                    placeRemainingParts();

                    % Check if the new parts placement matrix is a better
                    % placement (NOTICE - must estimate since puzzle solution is still known).
                    currentPartsMatScore = estimatePartsMatCorrectness();           
                    if (currentPartsMatScore > bestPartsMatScore)
                        bestPartsMatScore = currentPartsMatScore;
                        bestPartsMat = partsMat;
                    end
                end
            end
        end
    end

    % Main shifting function, activates the wanted shifting algorithm.
    function shiftJigsawParts()
        if (shiftMode >= 1)
            if (shiftMode == 1)
                % Shift largest segment to its best location.
                [partsMat] = shiftSegments(1,1);
            elseif (shiftMode >= 2)
                % Activate the segment and shift process, until
                % convergence.
                [partsMat] = convergeWithSegmentation();
            end
        end
    end

    % FLOATING PUZZLE  FUNCTIONS
    % %%%%%%%%%%%%%%%%%%%
    
    % Remove outer neighbors (when the floating is stopped)
    function  removeOuterNeighbors(axe)
        for i = 1:numOfParts
            if ((axe == 1 && (neighborVec(1, i) < 1 || neighborVec(1, i) > rows)) || ...
                   (axe == 2 && (neighborVec(2, i) < 1 || neighborVec(2, i) > cols)))
                neighborVec(1, i) = 0;
                neighborVec(2, i) = 0;
                neighborVec(3, i) = 0;
            end
        end
    end

    % Moving the parts matrix in a cyclic manner.
    function floatThePartsMatrix(deltaX, deltaY)
        
        if (deltaX == 1)
            partsMat(2 : rows, :) = partsMat(1 : rows - 1, :);
            partsMat(1, :) = 0;
        elseif (deltaX == -1)
            partsMat(1 : rows - 1, :) = partsMat(2 : rows, :);
            partsMat(rows, :) = 0;
        elseif (deltaY == 1)
            partsMat(:, 2 : cols) = partsMat(:, 1 : cols - 1);
            partsMat(:, 1) = 0;
        elseif (deltaY == -1)
            partsMat(:, 1 : cols - 1) = partsMat(:, 2 : cols );
            partsMat(:, cols) = 0;
        end
    end

    % Moving the neighbor's vectors elements.
    function floatTheNeighborVec(deltaX, deltaY)
        for i = 1:numOfParts
            if (neighborVec(1, i) ~= 0 || neighborVec(2, i) ~= 0 || neighborVec(3, i) ~= 0)
                neighborVec(1, i) = neighborVec(1, i) + deltaX;
                neighborVec(2, i) = neighborVec(2, i) + deltaY;
            end
        end
    end
    
    % Main function for the floating module
    function [x,y] = activateFloatingModule(x,y)
        deltaX = 0;
        deltaY = 0;
        
        % First we verify if indeed floating is needed, it is checked if
        % the neighbor is out of the puzzle frame.
        if  (x < 1)
            oppositeVec = partsMat(rows, :);
            axe = 1;
            deltaX = 1;
        elseif (x > rows)
            oppositeVec = partsMat(1, :);
            axe = 1;
            deltaX = -1;
        elseif (y< 1)
            oppositeVec = partsMat(:, cols);
            axe = 2;
            deltaY = 1;
        elseif (y > cols)
            oppositeVec = partsMat(:, 1);
            axe = 2;
            deltaY = -1;
        else
            return;
        end
        
        if (floatActiveInAxe(axe) == 0)
        	assert(false, 'Error in activateFloatingModule - floating was supposed to be active for the current axe.');
        end
        
        % From this point we know that a floating is needed, since the
        % previous else block contains a return command.
        
        % Checking if the floating is possible
        if all(oppositeVec == 0)
            
            % Performe the parts matrix floating
            floatThePartsMatrix(deltaX, deltaY);
            
            % Update the neighbor vec.
            floatTheNeighborVec(deltaX, deltaY);
            
            % Update the placement coordinates
            x = x + deltaX;
            y = y + deltaY;
        else 
            
            % Disable floating in the current axe.
            floatActiveInAxe(axe) = 0;
            
            % Removing outer neighbors in that axe.
             removeOuterNeighbors(axe);
             
             % Mark to the calling function that this placment should
             % not take place.
             x = -1;
             y = -1;
        end
    end
    
    % GREEDY ALGORITHM  FUNCTIONS
    % %%%%%%%%%%%%%%%%%%%%

    % Stores in the partsCompVal matrix the raw value of the parts
    % compatibility according to the selected compatibility function.
    function initializePartsCompatibility() 
        for i = 1:numOfParts - 1
            for j = i+1:numOfParts
                for l = 1:4
                    partsCompVal(i,j,l) = partsCompatibility(i, j, l, partsCorrectOrder);
                end
            end
        end
    end

    % Estimate the best seed, according to the compatibility values.
    function bestEstimatedSeed = selectBestSeedCandidate()
        % Counting for each part its estimated strength as a seed.
        for i = 1: numOfParts
            numOfbestBuddies = 0;
            
            for l = 1: 4
                bestBuddy = bestNeighbors(i, l);
                
                if (l == 1 || l == 3)
                    opposite = l + 1;
                else
                    opposite = l -  1;
                end
                
                if (bestNeighbors(bestBuddy, opposite) == i)
                    numOfbestBuddies = numOfbestBuddies + 1;
                end
            end
            
            bestBuddies(1, i) = i;
            bestBuddies(2, i) = numOfbestBuddies;
        end
        
        % Sort the best buddies vector according to the number of best
        % buddies a certain cell has.
        [B,IX] = sort(bestBuddies,2, 'descend');
        tempBestBuddies = [bestBuddies(1,IX(2,:));B(2,:)];
        bestBuddies = tempBestBuddies;
        
        % Return the first part in the sorted vector - the one with the
        % most best buddies.
        bestEstimatedSeed = bestBuddies(1);
    end

    % Remove a part from the unplaced parts vector and decrease the
    % unplaced counter.
    function removePartFromUnplaced(partIndex)
        if (partIndex > unplacedCounter)
            assert(false, 'Error in removePartFromUnplaced - partIndex is bigger then the unplacedCounter');
        else
            tempPart = unplacedParts(partIndex);
            unplacedParts(partIndex) = unplacedParts(unplacedCounter);
            unplacedParts(unplacedCounter) = tempPart;
            unplacedCounter = unplacedCounter - 1;
        end
    end

    % Adds the neighbors of the part at location (x,y) to the neighbors
    % vector.
    function  addPartNeighbors(x,y)
                
        if (x > 1 && partsMat(x-1,y) == 0) || (floatingPuzzleMode && floatActiveInAxe(1) && x == 1)
            addCellToNeighborsVec(x-1,y);
        end
        if (x < rows && partsMat(x+1,y) == 0) || (floatingPuzzleMode && floatActiveInAxe(1) && x == rows)
            addCellToNeighborsVec(x+1,y);
        end
        if (y > 1 && partsMat(x,y-1) == 0) || (floatingPuzzleMode && floatActiveInAxe(2) && y == 1)
            addCellToNeighborsVec(x,y-1);
        end
        if (y < cols && partsMat(x,y+1) == 0) || (floatingPuzzleMode && floatActiveInAxe(2) && y == cols)
            addCellToNeighborsVec(x,y+1);
        end
    end

    % Adds a new cell to the neighbors vector, or if already exsits in
    % the vector we update the number of known part neighbors.  Assume we 
    % can add this neighbor (valid x and y).
    function addCellToNeighborsVec(x,y)
        % Checks if already in the vector, if so we mark another known part
        % and return.
        for i = 1:numOfParts
            if (neighborVec(1, i) == x) &&(neighborVec(2, i) == y) 
                neighborVec(3, i) = neighborVec(3, i) + 1;
                return;
            end
        end
        
        % Since we did not find it in the vector, we add it.
        for i = 1:numOfParts
            % Checking if the current location is free (x = 0 is not enough!!!)
            if (neighborVec(1, i) == 0) && (neighborVec(2, i) == 0) && (neighborVec(3, i) == 0)
                neighborVec(1, i) = x;
                neighborVec(2, i) = y;
                neighborVec(3, i) = 1;
                break
            end
        end
    end

    % Removes a cell from the neighbors vector
    function removeCellFromNeighborsVec(x,y)
        for i = 1:numOfParts
            if (neighborVec(1, i) == x) &&(neighborVec(2, i) == y)
                neighborVec(1, i) = 0;
                neighborVec(2, i) = 0;
                neighborVec(3, i) = 0;
                return
            end
        end
        
        assert(false, 'Error in removeCellFromNeighborsVec - part argument is not in the neighborVec');
    end

    % Sort the neighbors vector in descending  order of known parts.
    function sortNeighborsByKnownParts()
        [B,IX] = sort(neighborVec,2, 'descend');
        tempNeighborVec = [neighborVec(1,IX(3,:));neighborVec(2,IX(3,:));B(3,:)];
        neighborVec = tempNeighborVec;
    end

    % Calculate the placement probability.
    function placementProbability =  calcPlacementProbability(x, y, partNdx)
        if (mod(greedyType,2) == 0)
            placementProbability = 1;
            if (x > 1 && y >= 1 && y <= cols && partsMat(x-1,y) ~= 0)
                placementProbability = placementProbability * confidence(unplacedParts(partNdx), partsMat(x-1,y), 3);
            end
            if (x < rows && y >= 1 && y <= cols && partsMat(x+1,y) ~= 0)
                placementProbability = placementProbability * confidence(unplacedParts(partNdx), partsMat(x+1,y), 4);
            end
            if (y > 1 && x >= 1 && x <=rows && partsMat(x,y-1) ~= 0)
                placementProbability = placementProbability * confidence(unplacedParts(partNdx), partsMat(x,y-1), 1);
            end
            if (y < cols && x >= 1 && x <=rows && partsMat(x,y+1) ~= 0)
                placementProbability = placementProbability * confidence(unplacedParts(partNdx), partsMat(x,y+1), 2);
            end
        elseif (mod(greedyType,2) == 1)
            neighbors = 0;
            placementProbability = 0;
            if (x > 1 && y >= 1 && y <= cols && partsMat(x-1,y) ~= 0)
                placementProbability = placementProbability + confidence(unplacedParts(partNdx), partsMat(x-1,y), 3);
                neighbors = neighbors + 1;
            end
            if (x < rows && y >= 1 && y <= cols && partsMat(x+1,y) ~= 0)
                placementProbability = placementProbability + confidence(unplacedParts(partNdx), partsMat(x+1,y), 4);
                neighbors = neighbors + 1;
            end
            if (y > 1 && x >= 1 && x <=rows && partsMat(x,y-1) ~= 0)
                placementProbability = placementProbability + confidence(unplacedParts(partNdx), partsMat(x,y-1), 1);
                neighbors = neighbors + 1;
            end
            if (y < cols && x >= 1 && x <=rows && partsMat(x,y+1) ~= 0)
                placementProbability = placementProbability + confidence(unplacedParts(partNdx), partsMat(x,y+1), 2);
                neighbors = neighbors + 1;
            end
            
            placementProbability = placementProbability / neighbors;
        else
             assert(false, 'Error in calcBestPlacementProbability - greedyType unknown!!');
        end
    end

    % Finds the best placement for a current cell, from the unplaced cells.
    function [placementProbability, placementNdx] = calcBestPlacementProbability(x, y)
        maxPlacementProb = -1;
        maxPlacementNdx = 0;
        
        for i = 1:unplacedCounter
             currPlacementProb =  calcPlacementProbability(x, y, i);
            
            if (currPlacementProb > maxPlacementProb)
                maxPlacementProb = currPlacementProb;
                maxPlacementNdx = i;
            end
        end

        placementProbability = maxPlacementProb;
        placementNdx = maxPlacementNdx;
    end

	% Find the next part to place using the 
    function [unplacedNdx, x, y] = narrowBestBuddiesPlacment(maxNumOfKnown)
        % Finding out how many cells with maxNumOfKnown neighbors.
        for cellCandidates = 1:numOfParts
            if (neighborVec(3, cellCandidates) < maxNumOfKnown)
                break;
            end
        end
        cellCandidates = cellCandidates - 1;
        
        % Narrow data structures.
        bestForUnplaced = zeros(2, unplacedCounter);
        bestForCell = zeros(2, cellCandidates);
        
        % Initialize the data structures.
        for i = 1:cellCandidates
            for j = 1:unplacedCounter
                currPlacementProb =  calcPlacementProbability(neighborVec(1,i), neighborVec(2,i), j);
                
                % Checking if this placment is best for the current part.
                if (currPlacementProb > bestForUnplaced(2, j))
                    bestForUnplaced(1, j) = i;
                    bestForUnplaced(2, j) = currPlacementProb;
                end
                
                % Checking if this part is best for the current placment.
                if (currPlacementProb > bestForCell(2, i))
                    bestForCell(1, i) = j;
                    bestForCell(2, i) = currPlacementProb;
                end
            end
        end
        
        % Finding the best buddies with the higest probability.
        unplacedNdx = 0;
        x = -1;
        y = -1;
        bestProbability = -1;
        
        for i = 1:cellCandidates
            % If they are  best buddies. We check if (bestForCell(1, i) > 0)
            % since it might happen if the probability was alwasys zero.
            if (bestForCell(1, i) > 0) && (i == bestForUnplaced(1, bestForCell(1, i)))
                
                if (mod(greedyType,2) == 0)
                    currentProb = bestForCell(1, i)  *  bestForUnplaced(1, bestForCell(1, i));
                elseif (mod(greedyType,2) == 1)
                    currentProb = (bestForCell(1, i) + bestForUnplaced(1, bestForCell(1, i))) / 2;
                end
                
                if (currentProb > bestProbability)
                    unplacedNdx = bestForCell(1, i);
                    x = neighborVec(1,i);
                    y = neighborVec(2,i);
                    bestProbability = currentProb;
                end
            end
        end
    end

    % Looks at the neighbors vector and at the unplaced parts, and decides
    % what is the best pair (cell and part).
    function [unplacedNdx, x, y] = findBestPlacement()
        unplacedNdx = 0;
        sortNeighborsByKnownParts();
        
        % Since the neighbors are sorted according to the known field the
        % highest value is in the first cell.
        maxNumOfKnown = neighborVec(3,1);
        
        if (maxNumOfKnown == 0)
            assert(false, 'Error in findBestPlacement - maxNumOfKnown can not be zero!!');
        end
        
        maxPlacementProbability = -1;
        i = 1;
        
        % In this case activate the narrowed best buddies.
        if (greedyType ~= 0) && (greedyType ~= 1)
            [unplacedNdx, x, y] = narrowBestBuddiesPlacment(maxNumOfKnown);
            
            % The narrow best buddies might not find such a placement, if
            % so we will need to resorte to a basic placement.
            if (unplacedNdx ~= 0)
                return
            end
        end
        
        % Going over neighbors that are connected to the max amount of
        % known cells and choosing the placement from them. We do so
        % because the more known cells to  a neighbor, the more chance we
        % will select the right choice.
        while (neighborVec(3,i) == maxNumOfKnown)
                [placementProbability, placementNdx] = calcBestPlacementProbability(neighborVec(1,i), neighborVec(2,i));

                if (placementProbability > maxPlacementProbability)
                    maxPlacementProbability  = placementProbability;
                    x = neighborVec(1,i);
                    y = neighborVec(2,i);
                    unplacedNdx = placementNdx;
                end

                i = i + 1;
        end
    end

    % Main function that places the remaining parts in a greedy manner.
    function placeRemainingParts()
        runIterations = 0;
        
        % As long as there are parts to place
        while (unplacedCounter > 0)
            % When debug mode is activated, show the user the current
            % state of the solution.
            if (debug) 
                if (runIterations <= 0)
                    % Show the current placment to the user. This adds a
                    % movie frame if needed.
                    displayCurrentPlacement();
                    
                    switch debug
                        case 1 % manual no movie
                            runIterations = input('Enter num of placments to place, or enter for one: ');
                        case 2 % manual with movie
                            % No prompt to allow a simple copy paste from the
                            % command window. Thus, reproducing the movie
                            % easliy
                            runIterations = input('');
                        case {3, 4} % automatic
                            runIterations = floor(numOfParts / 20);
                            pause(1/framePerSecond);
                        otherwise
                            assert(false, 'Error in placeRemainingParts debug value is invalid');
                    end
                end
                
                if (isempty(runIterations))
                    runIterations = 0;
                elseif (runIterations > 0)
                    runIterations = runIterations - 1;
                end
            end
        
            % Finds the best index and the best part in which to place the
            % part.
            [unplacedNdx, x, y] = findBestPlacement();
            
            % Checking if the floating puzzle should be activated.
            if (floatingPuzzleMode == 1)
                [x,y] = activateFloatingModule(x,y);
                
                % A flag to skip the current placement.
                if (x == -1 || y == -1)
                    continue
                end
            end
            
            % Place the best part. Update all the relevent data structures
            partsMat(x, y) = unplacedParts(unplacedNdx);
            removePartFromUnplaced(unplacedNdx);
            removeCellFromNeighborsVec(x, y);
            addPartNeighbors(x, y);
        end
        
        if (debug) 
            % Create a frame in the movie of the current iteration after all
            % the parts have been placed.
            displayCurrentPlacement();
        end
    end

    % Main function for placing the parts
    function greedyPlacingAlgorithm()
        % Place inital part and initialize data structures
        partsMat(floor(rows/2), floor(cols/2)) = unplacedParts(firstPartToPlace);
        removePartFromUnplaced(firstPartToPlace);
        addPartNeighbors(floor(rows/2) , floor(cols/2));
        
        % Placing the parts in a greedy manner
        placeRemainingParts();
        
        % If needed save the current result
        if (outputMode == 2)
            % Create the  parts order from the parts placing matrix
            partsOrder = convertPartsMatToVec();
            [onlyGreedyDirectCompVal, onlyGreedyNeighborCompVal] = jigsawComparison(partsOrder);
            directCompVal = onlyGreedyDirectCompVal;
            neighborCompVal = onlyGreedyNeighborCompVal;
            onlyGreedyAlgoImage = buildOutputImage(partsOrder);
            displayImage(onlyGreedyAlgoImage, 1, 1); % 1 = save
        end
    end

    % Jigsaw puzzle comparison functions (Performance metrics)
    % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Generic comparison function
    function [directCompVal, neighborCompVal] = jigsawComparison(order)
        directCompVal = directComparison(order);
        neighborCompVal = neighborComparison(order);
    end

    % Calculate the direct comparison of a parts order, with thier
    % correct order
    function [compVal] = directComparison(order)
        compVal = length(find((order - partsCorrectOrder) == 0)) / numOfParts;
    end

    % Calculate the neighbor comparison of a parts order
    function [compVal] = neighborComparison(order)
       actualMat = zeros(rows + 2,cols + 2);
       expectedMat = zeros(rows + 2,cols + 2);
       hits = 0;
       
       % Filling the matrices
        for i = 1:rows
            for j = 1:cols
                current = (i - 1) * cols + j;
                actualMat(i + 1,j + 1) = order(current);
                expectedMat(i + 1,j + 1) = current;
            end
        end
        
        for i = 1:numOfParts
            % Finding part i in the actual and expected matrices
            actNdx = find(actualMat == i);
            expNdx = find(expectedMat == i);
            
            % Transfer from indexes to (x,y) coordinate
            x1 = mod(actNdx - 1, rows + 2)  + 1;
            y1 = floor((actNdx - 1) / (rows + 2)) + 1;
            x2 = mod(expNdx - 1, rows + 2)  + 1;
            y2 = floor((expNdx - 1) / (rows + 2)) + 1;

            % Comparing the neighbors.
            for dx = -1:1
                for dy = -1:1
                    % This condition will leave us with the neighbors.
                    if (dx * dy==0 && dx ~= dy)
                        hits = hits + (actualMat(x1+dx,y1+dy) == expectedMat(x2+dx,y2+dy));
                    end
                end
            end
        end

        compVal = hits / (numOfParts * 4);
    end

    %  Calculate the correctness according to the difference of each pixel
    %  from the corresponding pixel in the original image (Resemblance metric).
    function [compVal] = resemblanceComparison(order)
        compVal = 0;
        for index = 1 : numOfParts
            partDiff = abs(newColorSchemePartsArray(:,:,:, index) - newColorSchemePartsArray(:,:,:, order(index)));
            compVal = compVal + sum(sum(sum(partDiff)));
        end
    end

    % Jigsaw puzzle correctness estimation functions (Estimation metrics)
    % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Estimate the parts placment matrix correctness. Use this function and
    % not the comparison functions (Performance metrics) when the puzzle
    % solution is still known.
    function [estimatedVal] = estimatePartsMatCorrectness()
        if (estimationFunc == 0)
            estimatedVal = estimateCorrectnessByBestBuddy();
        elseif (estimationFunc == 1)
            estimatedVal = estimateCorrectnessAverageOfProb();
        end
    end

    % Estimate the correctness according to the Best Buddy method,
    % calculate the ratio between Best Buddy connection and all possible
    % connections.
    function [estimatedVal] = estimateCorrectnessByBestBuddy()
        estimatedVal = 0;
        
        for i = 1:(rows - 1)
            for j = 1:cols
                estimatedVal = estimatedVal + ...
                                                     (bestNeighbors(partsMat(i,j), 4) == partsMat(i+1,j) && ...
                                                      bestNeighbors(partsMat(i+1,j), 3) == partsMat(i,j));
            end
        end
        
        for j = 1:(cols - 1)
            for i = 1:rows
                estimatedVal = estimatedVal + ...
                                                     (bestNeighbors(partsMat(i,j), 2) == partsMat(i,j+1) && ...
                                                      bestNeighbors(partsMat(i,j+1), 1) == partsMat(i,j));
            end
        end
        
        estimatedVal = estimatedVal /((rows - 1) * cols + (cols - 1) * rows);
    end

    % Estimate correctnes by the average of the sum of probabilities.
    function [estimatedVal] = estimateCorrectnessAverageOfProb()
        estimatedVal = 0;
        
        for i = 1:(rows - 1)
            for j = 1:cols
                estimatedVal = estimatedVal + confidence(i,j,4) + confidence(i+1,j,3);
            end
        end
        
        for j = 1:(cols - 1)
            for i = 1:rows
                estimatedVal = estimatedVal + confidence(i,j,2) + confidence(i,j+1,1);
            end
        end
        
        estimatedVal = estimatedVal /((rows - 1) * cols * 2 + (cols - 1) * rows * 2);
    end

	% Parts compatibility functions
    % %%%%%%%%%%%%%%%%%%%%%%
    
    % Creates the confidence zero iteration guess.
    function initializeConfidence() 
        for i = 1:numOfParts
            for j = 1:numOfParts
                if (i ~= j)
                    for l = 1:4
                        if (i < j)
                            confidence(i,j,l) = partsCompVal(i, j, l);
                        else
                            % Since i > j then it means due to symmetry
                            % that we can use opposite label of a calculate
                            % value.
                            if (l == 1 || l == 3)
                                confidence(i,j,l) = partsCompVal(j, i, l+1);
                            else
                                confidence(i,j,l) = partsCompVal(j, i, l-1);
                            end
                        end
                    end
                end
            end
        end
        
        % Calc probability.
        for i = 1:numOfParts
            for l = 1:4
                if (normalMode == 1)
                    % First method - linear according to the max value.
                    maxVal = max(confidence(i,:,l));
                    confidence(i,:,l) = (maxVal - confidence(i,:,l)) / maxVal;
                elseif (normalMode == 2)
                    % Second method as used by Cho et al.
                    tempCon =  confidence(i,:,l);
                    tempCon = sort(tempCon);
                    smallest = tempCon(1);
                    secondSmallest = tempCon(2);
                    confidence(i,:,l) = exp(-confidence(i,:,l)/(2*(smallest-secondSmallest)^2));
                elseif (normalMode == 3)
                    % Third method - exp according to median value.
                    medianVal = median(confidence(i,:,l));
                    confidence(i,:,l) = exp(-confidence(i,:,l)/(medianVal));
                elseif (normalMode == 4)
                    % Forth method - exp according to first quartile value. 
                    sortedConfidence = sort(confidence(i,:,l));
                    firstQuartile = median(sortedConfidence(1:(ceil(numOfParts/2))));

                    % Handling the case when firstQuartile is zero, it would
                    % lead to NaN values since 0/0 in matlab can not be
                    % computed. (BUG FIX)
                    if (firstQuartile == 0) 
                        % This magic number is the smallest double value.
                        firstQuartile = 2.22507e-308; 
                    end

                    confidence(i,:,l) = exp(-confidence(i,:,l)/(firstQuartile));
                end
            end
        end
    end
    
    % Generic compatibility function.
    function [compVal] = partsCompatibility(first, second, relation, order)
        if (length(compatibilityFunc ) > 1) || (compatibilityFunc <= 1)
            compVal = dissimilarityBasedCompatibility(first, second, relation, order);
        elseif ((compatibilityFunc >= 2) && (compatibilityFunc <= 4) || ...
                           compatibilityFunc == 22 || compatibilityFunc >= 44)
            compVal = differenceBasedCompatibility(first, second, relation, order);
        else
            assert(false, 'Error in partsCompatibility - compatibility function not supported');
        end
    end

    % Calculate the dissimilarity  based compatibility
    % between two parts according to thier realtion and a defining order.
    function [compVal]  = dissimilarityBasedCompatibility(first, second, relation, order)
        firstPart = newColorSchemePartsArray(:,:,:, order(first)); 
        secondPart = newColorSchemePartsArray(:,:,:, order(second));
        
        switch relation
            case 1 % 'left'
                firstVec = firstPart(:,1,:);
                secondVec = secondPart(:,partSize,:);
            case 2 % 'right'
                firstVec = firstPart(:,partSize,:);
                secondVec = secondPart(:,1,:);
            case 3 % 'up'
                firstVec = firstPart(1,:,:);
                secondVec = secondPart(partSize,:,:);
            case 4 % 'down'
                firstVec = firstPart(partSize,:,:);
                secondVec = secondPart(1,:,:);
            otherwise
                assert(false, 'Error in dissimilarityBasedCompatibility relation must be: left,right,up or down (1-4)');
        end
        
        if (length(compatibilityFunc ) > 1)
            p = compatibilityFunc(1);
            q = compatibilityFunc(2);
            compVal = sum(sum((abs(firstVec - secondVec)).^p)).^(q/p);
        elseif (compatibilityFunc == 0)  
            compVal = sum(sum((firstVec - secondVec).^2));
        elseif (compatibilityFunc == 1)
            compVal = sum(sum(sqrt(abs(firstVec - secondVec))));
        elseif (compatibilityFunc < 0)
            p = abs(compatibilityFunc);
            compVal = sum(sum((abs(firstVec - secondVec)).^p)).^(1/p);
        end
    end

    % Calculate the dissimilarity according to a certain difference
    % method (Determined by the compatibilityFunc parameter).
    function [compVal] = differenceBasedCompatibility(first, second, relation, order)
        firstPart = newColorSchemePartsArray(:,:,:, order(first)); 
        secondPart = newColorSchemePartsArray(:,:,:, order(second));
        
        firstVec = zeros(3,partSize,3);
        secondVec = zeros(3,partSize,3);
        
        % The extraction of data is common to all difference methods
        % We want to have two matrices, that in row 1 have the line that is
        % next to the border line, in row 2 the border line. 
        switch relation
            case 1 % 'left'
                firstVec = transpose3D(firstPart,2,1);
                secondVec = transpose3D(secondPart,partSize - 1,partSize);
            case 2 % 'right'
                firstVec = transpose3D(firstPart,partSize - 1,partSize);
                secondVec = transpose3D(secondPart,2,1);
            case 3 % 'up'
                firstVec(1:2,:,:) = firstPart(2:-1:1,:,:);
                secondVec(1:2,:,:) = secondPart(partSize - 1:partSize,:,:);
            case 4 % 'down'
                firstVec(1:2,:,:) = firstPart(partSize - 1:partSize,:,:);
                secondVec(1:2,:,:) = secondPart(2:-1:1,:,:);
            otherwise
                assert(false, 'Error in backwardDifferenceBasedCompatibility relation must be: left,right,up or down (1-4)');
        end
        
        % The different methods
        if (compatibilityFunc == 2) ||  (compatibilityFunc == 22)
            % First method (backward difference):we estimate the value of
            % the next pixel and compare it with the actual value (We do so 
            % from both sides of the connection).
            
            % In row 3 we will place the approximation of the next value for both vectors.
            firstVec(3,:,:) = firstVec(2,:,:) + (firstVec(2,:,:) - firstVec(1,:,:));
            secondVec(3,:,:) = secondVec(2,:,:) + (secondVec(2,:,:) - secondVec(1,:,:));
        
            % For both vectors we will calculate the dissimilarity between the
            % actual value and the approximation.
            if (compatibilityFunc == 2)
                compVal = sum(sum(sum((firstVec(3,:,:) - secondVec(2,:,:)).^2))) + ...
                                       sum(sum(sum((firstVec(2,:,:) - secondVec(3,:,:)).^2)));
            else
                p = 0.3;
                q = 1/32;
                compVal = sum(sum(sum((abs(firstVec(3,:,:) - secondVec(2,:,:))).^p))).^(q/p) + ...
                                       sum(sum(sum((abs(firstVec(2,:,:) - secondVec(3,:,:))).^p))).^(q/p);
            end
        elseif (compatibilityFunc == 3)
            % Second method (backward difference average):we estimate the
            % value of differences from both sides, calc an average and
            % compare it with the value of the estimated result.
            
            % In row 3 we will place the  value of the backward difference
            % for each vector.
            firstVec(3,:,:) = abs(firstVec(2,:,:) - firstVec(1,:,:));
            secondVec(3,:,:) = abs(secondVec(2,:,:) - secondVec(1,:,:));

            compVal = sum(sum(sum(((firstVec(3,:,:) + secondVec(3,:,:)) / 2 - abs(firstVec(2,:,:) - secondVec(2,:,:))).^2)));
        elseif (compatibilityFunc == 4) || (compatibilityFunc == 44)
            % Third method (central difference): calculate the estimated
            % central difference at for both parts, according to
            % information from the other part (f(x+h) - from other vec)
            
            firstVec(3,:,:) = (secondVec(2,:,:) - firstVec(1,:,:)) / 2 + firstVec(2,:,:);
            secondVec(3,:,:) = (secondVec(1,:,:) - firstVec(2,:,:)) / 2 + secondVec(2,:,:);
            
%             if (compatibilityFunc == 4)
%                 compVal = sum(sum(sum((firstVec(3,:,:) - secondVec(3,:,:)).^2)));
%             else
%                 compVal = sum(sum(sum(sqrt(abs(firstVec(3,:,:) - secondVec(3,:,:))))));
%             end

            % For both vectors we will calculate the dissimilarity between the
            % actual value and the approximation.
            if (compatibilityFunc == 4)
                compVal = sum(sum(sum((firstVec(3,:,:) - secondVec(2,:,:)).^2))) + ...
                                       sum(sum(sum((firstVec(2,:,:) - secondVec(3,:,:)).^2)));
            else
                 compVal = sum(sum(sum(sqrt(abs(firstVec(3,:,:) - secondVec(2,:,:)))))) + ...
                                        sum(sum(sum(sqrt(abs(firstVec(2,:,:) - secondVec(3,:,:))))));
            end
        end
    end

    % Calculate the transpose of a 3 dim matrix. Using a regualr transpose
    % on each index of the thrid element.
    function [transVec] = transpose3D(part, colStart, colEnd)
        transVec = zeros(3,partSize,4);
        
        for i = 1:3
            % Notice the transpose ' keyword at the end of the statement.
            transVec(1:2,:,i) = part(:,colStart:colEnd-colStart:colEnd,i)';
        end
    end
 
	% Utility  functions
    % %%%%%%%%%%%%%
    
    % Initialize default values.
    function initDefaultValues(numOfArgsForMainFunc)
        if (numOfArgsForMainFunc <= 2)
            partSize = 28;
        end
        if (numOfArgsForMainFunc <= 3)
            compatibilityFunc = 22;
        end
        if (numOfArgsForMainFunc <= 4)
            shiftMode = 2;
        end
        if (numOfArgsForMainFunc <=5)
            colorScheme = 1;
        end
        if (numOfArgsForMainFunc <=6)
            greedyType = 3;
        end
        if (numOfArgsForMainFunc <=7)
            estimationFunc = 0;
        end
        if (numOfArgsForMainFunc <=8)
            outputMode = 1;
        end
        if  (numOfArgsForMainFunc <=10)
            normalMode = 4;
        end
        if (numOfArgsForMainFunc <=11)
            resPath = 'C:\temp\';
        end
        if (numOfArgsForMainFunc <=12)
            debug = 0;
        end
        
        % Fixing the resPath if needed.
        if (resPath(length(resPath)) ~= '\')
            resPath = strcat(resPath, '\');
        end
    end
 
    % Initialize output variables to have values that indicate that the
    % solver was not activated.
    function setOutputToSolverNotActivated()
        directCompVal = -1;
        neighborCompVal = -1;
        estimatedBestBuddiesCorrectness = -1;
        estimatedAverageOfProbCorrectness = -1;
        solutionPartsOrder = -1;
        segmentsAccuracy = -1;
        numOfIterations = -1;
    end
 
    % Cuts the images into parts.
    function cutImageToParts()
        for index = 1 : numOfParts
            rowStartIndex = (ceil(index / cols)  - 1) * partSize + 1;
            rowEndIndex = rowStartIndex + (partSize -  1);
            colStartIndex = mod(index - 1, cols)  * partSize + 1;
            colEndIndex = colStartIndex + (partSize -  1);
            newColorSchemePartsArray(:,:,:, index) = imageWithNewColorScheme(rowStartIndex :rowEndIndex, colStartIndex :colEndIndex, :);
            rgbPartsArray(:,:,:, index) = image(rowStartIndex :rowEndIndex, colStartIndex :colEndIndex, :);
        end
    end

    % Converts the image to a user defined color scheme.
    function [convertedImage] = convertToUserSelectedColorScheme()
        if (colorScheme == 0)
            convertedImage = image;
        elseif (colorScheme == 1)
            cformRgbToLab = makecform('srgb2lab');
            convertedImage = double(applycform(image, cformRgbToLab));
        elseif (colorScheme == 2)
            convertedImage = rgb2hsv(image);
        else
             assert(false, strcat('Color scheme:',num2str(colorScheme),' is not supported.'));
        end
    end

    % Converts the parts matrix to a parts vector
    function [partsOrder] = convertPartsMatToVec()
        partsOrder = zeros(1, numOfParts);
        
        for i = 1:rows
            for j = 1:cols
                partsOrder((i - 1) * cols + j) = partsMat(i,j);
            end
        end
    end

    % Builds the ouptut image according to the parts order
    function [reconstructedImage] = buildOutputImage(order)
        reconstructedImage = zeros(imageSize);
        
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

    % Display an image according to the format.
    function displayImage(imageToShow, shouldSave, showCaption)
        fileExtension = lower(I(length(I)-3:length(I)));
        
        if (showCaption)
            figure('Name', strcat('Jigsaw Solver solution for image: ', name, ext, ...
                        ', seed=', num2str(firstPartToPlace), ', num of parts=', num2str(numOfParts), ...
                        '. Results: direct=', num2str(100*directCompVal), '%, neighbor=', num2str(100*neighborCompVal), '%.'), ...
                        'NumberTitle','off');
        end
        
        if (strcmp(fileExtension ,'.jpg'))
                convertedImage = uint8(imageToShow);
        elseif (strcmp(fileExtension ,'.png'))
                convertedImage = uint16(imageToShow);
        else
            fprintf('Sorry, but we dont know how to display %g files', fileExtension);
            return
        end
        
        if (debug && ~shouldSave)
            h = imshow(convertedImage, 'InitialMagnification', 'fit');
            set(1, 'WindowStyle', 'docked');
        else
            h = imshow(convertedImage, 'InitialMagnification', 100);
            set(1, 'WindowStyle', 'Normal');
        end
        
        if (shouldSave)
            axis image
            axis off
            saveas(h, strcat(resPath,imageName,'(', num2str(figNum), ').fig'));
            figNum = figNum + 1;
        end
    end

    % Display the current placment, if debug mode is set to create a movie
    % then a frame is added to the avi object movie.
    function displayCurrentPlacement()
        partsOrder = convertPartsMatToVec();
        partialImage = buildOutputImage(partsOrder);
        displayImage(partialImage, 0, 0); % 0 = don't save

        % Create a frame in the movie
        if (debug == 2)  || (debug == 4) 
            frame = getframe(gca);
            aviobj = addframe(aviobj,frame);
        end
    end

    % Create a visualization movie that illustrates how the solver shuld
    % work.
    function createVisualizationMovie()
        debug = 0;
        partsOrder = randperm(numOfParts);

        % Creating and saving the movie.
        NUM_OF_FRAMES = 200; 
        visAviObj = avifile(strcat(resPath,'JigsawSolverVisualization.avi'),'fps',24,'compression', 'None');
        
        % The first frame
        outputImage = buildVisualizationFrame(partsOrder, 0, NUM_OF_FRAMES);
        
         % Saving the first frame as an image (mixed image.
        fileExtension = lower(I(length(I)-2:length(I)));
        if (strcmp(fileExtension ,'jpg'))
                mixedImage = uint8(outputImage);
        elseif (strcmp(fileExtension ,'png'))
                mixedImage = uint16(outputImage);
        else
            fprintf('Sorry, but we dont know how to display %g files', fileExtension);
            return
        end
        imwrite(mixedImage, strcat(resPath,'JigsawSolverVisualization','.',fileExtension), fileExtension);
        
        displayImage(outputImage, 0, 0); % 0,0 = don't save and this will prevent the movie from opening in many windows.
        frame = getframe(gca);
        for frameNdx = 1:24
            visAviObj = addframe(visAviObj,frame);
        end

        for frameNdx = 0:NUM_OF_FRAMES
            outputImage = buildVisualizationFrame(partsOrder, frameNdx, NUM_OF_FRAMES);
            displayImage(outputImage, 0, 0); % 0 = don't save and this will prevent the movie from opening in many windows.
            frame = getframe(gca);
            visAviObj = addframe(visAviObj,frame);
        end

        for frameNdx = 1:24
            visAviObj = addframe(visAviObj,frame);
        end

        aviobj = close(visAviObj);
        fprintf('Finished visualization movie\n');
    end

    % Builds a visualization movie frame, by calculating each parts
    % location according to the index argument. This argument represents
    % number of frame to create
    function [reconstructedImage] = buildVisualizationFrame(order, index, frames)
        reconstructedImage = zeros(imageSize);
        
         for i = 1 : numOfParts
             startRowStartIndex = (ceil(i / cols)  - 1) * partSize + 1;
             startColStartIndex = mod(i - 1, cols)  * partSize + 1;
             
             endRowStartIndex = (ceil(order(i) / cols)  - 1) * partSize + 1;
             endColStartIndex = mod(order(i) - 1, cols)  * partSize + 1;
            
             rowStartIndex = floor(startRowStartIndex * (index / frames) + endRowStartIndex * ((frames - index) / frames));
             rowEndIndex = rowStartIndex + (partSize -  1);
             colStartIndex = floor(startColStartIndex * (index / frames) + endColStartIndex * ((frames - index) / frames));
             colEndIndex = colStartIndex + (partSize -  1);
             
             reconstructedImage(rowStartIndex :rowEndIndex, colStartIndex :colEndIndex, :) = ...
             	rgbPartsArray(:,:,:, i);
         end
    end

    % RELAXATION LABELING FUNCTIONS - NOT USED IN CVPR 2011 SOLVER
    % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Finds out for the first and second connection the common part and the
    % uncommon parts
    function [commonPart, fnc, snc] = findCommon(f1,f2,s1,s2)
        if (f1 == s1)
            commonPart = f1;
            fnc = f2;
            snc = s2;
        elseif (f1 == s2)
            commonPart = f1;
            fnc = f2;
            snc = s1;
        elseif (f2 == s1)
            commonPart = f2;
            fnc = f1;
            snc = s2;
        elseif (f2 == s2)
            commonPart = f2;
            fnc = f1;
            snc = s1;
        else
            assert(false, 'Error in findCommon - no common part in input.');
        end
    end

    % Finds the relation
    function [relationToCommon] = relationToCommon(commonPart, notCommon, label)
        relationToCommon = label;
        
        % If the common part has a bigger index, then the relation is the
        % opposite one.
        if (commonPart > notCommon)
            if (label == 2 || label == 4)
                relationToCommon = relationToCommon - 1;
            else
                relationToCommon = relationToCommon + 1;
            end
        end
    end

    % Returns the opposite realtion.
    function [oppositeRelation] = oppositeRelation(relation)
        if (relation == 1 || relation == 3)
            oppositeRelation = relation + 1;
        else
            oppositeRelation = relation - 1;
        end
    end

    % Checks for max multiplication of the compatiblity between a part and
    % the two input parts, according to the specified relations.
    function [maxCompVal] = bestPartCompatibility(commonPart, fnc, firstRelationToCommon, snc, secondRelationToCommon)
        maxCompVal = 0;
        
        if (firstRelationToCommon == secondRelationToCommon)
            assert(false, 'Error in bestPartCompatibility - relation cant be the same');
        end
        
        firstRelationToOther = secondRelationToCommon;
        secondRelationToOther = firstRelationToCommon;
                
        % Goes over the parts and finds the part that is most likely to
        % enter between fnc and snc (which is not common) - and return it's
        % probability.
        for i = 1:numOfParts
            % The part must not be one of three parameter parts.
            if (i ~= commonPart && i~= fnc && i~=snc)
                currentCompVal = 10 * confidence(fnc,i,firstRelationToOther) * confidence(i,fnc,oppositeRelation(firstRelationToOther)) * ...
                                                                  confidence(snc,i,secondRelationToOther) * confidence(i,snc,oppositeRelation(secondRelationToOther));
                
                if (currentCompVal > maxCompVal)
                    maxCompVal = currentCompVal;
                end
            end
        end
    end
    
    % Relaxation labeling compatibility function [r(i,j,Li,Lj)]
    % Assumes that f1 < f2 and s1 < s2.
    function [compVal] = compatibility(f1, f2, s1, s2, fl, sl)
        compVal = 0;
        
        % First we check if these are different parts.
        if (f1 ~= s1 && f1 ~= s2 && f2 ~= s1 && f2 ~= s2)
            % Do nothing: compVal = 0
        % Check if these are the same parts
        elseif (f1 == s1 && f2 == s2)
            % If the label is different then they conflict
            if (fl ~=  sl)
                compVal = -1;
            end
            % Else if same labels and parts  compVal = 0
        else
            % Now we know that exactly one of the parts is shared
            
            if (fl ==  5 || sl == 5) % 5 == 'no_relation'
                % Do nothing: compVal = 0
            else
                [commonPart, fnc, snc] = findCommon(f1,f2,s1,s2);
                firstRelationToCommon = relationToCommon(commonPart,fnc,fl);
                secondRelationToCommon = relationToCommon(commonPart,snc,sl);
                
                % If they mark the same connection with the common part, this
                % is a collision
                if (firstRelationToCommon == secondRelationToCommon)
                    compVal = -1;
                % "<=2" means left (1) or right(2).
                % ">=3" means up (3) or down (4).
                elseif ((firstRelationToCommon <= 2 && secondRelationToCommon <= 2) || ...
                                  (firstRelationToCommon >= 3 && secondRelationToCommon >= 3))
                    % In this case the parts are in opposite positions, we
                    % do nothing: compVal = 0
                else
                    % In this case there is a another part which is another
                    % common part. We check if there is a part that fits in
                    % well. If so this will be the only point where compVal
                    % will be grater then zero!
                    compVal = bestPartCompatibility(commonPart, fnc, firstRelationToCommon, snc, secondRelationToCommon);
                end
            end
        end
    end

    % Relaxation labeling support function
    function [supVal] = support(f1,f2,fl)
        supVal = 0;
        
        for i = 1:numOfParts - 1
             for j = i+1:numOfParts
                for l = 1:4
                    if (l == 1 || l == 3)
                        supVal = supVal + compatibility(f1,f2,i,j,fl,l) * confidence(i,j,l) * confidence(j,i,l + 1);
                    else
                        supVal = supVal + compatibility(f1,f2,i,j,fl,l) * confidence(i,j,l) * confidence(j,i,l - 1);
                    end
                end
            end
        end
    end

    % UNIT TESTS FUNCTIONS
    % %%%%%%%%%%%%%%%
    
    % Basic test function
    function test(cond, msg)
        assert(cond, msg);
        testsPassed = testsPassed + 1;
    end

    % Compare two vectors
    function [bool] = sameVector(vec1, vec2)
        bool = (length(vec1) == length(vec2)) && (sum(vec1 == vec2) == length(vec1));
    end

    % Checks if the compatibility function found the actual neighbor with
    % the highest probability to be the parts neighbor with that relation.
    function [hit] = wasCompatibilityFunctionCorrect(part, actualNeighbor, relation)
        hit = 0;
        partsVec = partsCompVal(part,:,relation);
        minNdxVec = find(partsVec==min(partsVec));
        
        if ((length(minNdxVec) == 1) && minNdxVec == actualNeighbor)
            hit = 1;
        end
    end

    % Checks the parts compatibility functions
    function  testPartsCompatibilityFunctions()
        %hitsSum = 0;
        numOfImages = 20;
        
        compatibilityStrength = zeros(1, numOfImages);
        
         [pathstr, name, ext] = fileparts(I);
        
        for imageIndex = 1:numOfImages
             hits = 0;
            
            % Reading the image and converting to the new color scheme.
            path = strcat(pathstr, '\', num2str(imageIndex), ext);
            image = imread(path);
            imageWithNewColorScheme = convertToUserSelectedColorScheme();

            image = double(image);
            imageSize = size(image);

            % Calculate a parts data
            rows = imageSize(1) / partSize;
            cols = imageSize(2) / partSize;
            numOfParts = rows * cols;
            partsCorrectOrder = 1:numOfParts;

            % Creating parts array that contain the puzzle parts
            newColorSchemePartsArray =  zeros(partSize, partSize, 3, numOfParts);
            rgbPartsArray =  zeros(partSize, partSize, 3, numOfParts);

            % Splits the image into parts.
            cutImageToParts();
            
            % Initialize parts compatibility, unlike when solving the jigsaw
            % we will fill the entire partsCompVal matrix.
            partsCompVal = zeros(numOfParts,numOfParts,4);
            initializePartsCompatibility();
            for i = 1:numOfParts
                for j = i:numOfParts
                    for l = 1:4
                        if (i == j)
                            partsCompVal(i,j,l) = bitmax;
                        else
                            if (l==1 || l==3)
                                partsCompVal(j,i,l) = partsCompVal(i, j, l+1);
                            else
                                partsCompVal(j,i,l) = partsCompVal(i, j, l-1);
                            end
                        end
                    end
                end
            end

            % Creating the expected result matrix so we will be able to
            % know a parts neighbors.
            partsExpMat = zeros(rows,cols);
            for i = 1:rows
                for j = 1:cols
                    partsExpMat(i,j) = (i - 1) * cols + j;
                end
            end
            
            % Going over the connections
            for i = 1:rows
                for j = 1:cols
                    if (i > 1) 
                        hits = hits + wasCompatibilityFunctionCorrect(partsExpMat(i,j), partsExpMat(i-1,j), 3);
                    end
                    if (i<rows)
                        hits = hits + wasCompatibilityFunctionCorrect(partsExpMat(i,j), partsExpMat(i+1,j), 4);
                    end
                    if (j > 1)
                        hits = hits + wasCompatibilityFunctionCorrect(partsExpMat(i,j), partsExpMat(i,j-1), 1);
                    end
                    if (j<cols)
                        hits = hits + wasCompatibilityFunctionCorrect(partsExpMat(i,j), partsExpMat(i,j+1), 2);
                    end
                end
            end
            
            %hitsSum = hitsSum + hits;
            
            % We count each connection (edge) twice, since we checked from each part's point of view.
            compatibilityStrength(imageIndex) = (hits /  (2 * (rows * (cols - 1) + (rows - 1) * cols)));
        end
        
        compatibilityStrength
        %accuracy = hitsSum / (2 *  numOfImages * (rows * (cols - 1) + (rows - 1) * cols));
        fprintf('Compatibility function %s\nColor %g\nPart size %g\nAccuracy %g%%\nAccuracy variance %g%%\n',  ... 
                            num2str(compatibilityFunc), colorScheme, partSize, ...
                            mean(compatibilityStrength)*100, var(compatibilityStrength)*100);
        
        % This is not a proper usage of the return values. When 
        % runTests == 1, directCompVal = Combatibility Accuracy,
        % neighborCompVal = Combatibility Accuracy variance.
        directCompVal =  mean(compatibilityStrength) * 100;
        neighborCompVal= var(compatibilityStrength) * 100;
    end

    % Calc best buddies and store the result in a local variable.
    function calcBestBuddiesForAllImages()
        numOfImages = 20;
        
        bestBuddiesForAllImages = zeros(2 * numOfImages, 432);
        
        for imageIndex = 1:numOfImages
            bestBuddies = zeros(2, numOfParts);
            
            % Reading the image and converting to the new color scheme.
            path = strcat('C:\temp\imData\', ...
                                            num2str(imageIndex), '.png');
            image = imread(path);
            imageWithNewColorScheme = convertToUserSelectedColorScheme();

            image = double(image);
            imageSize = size(image);

            % Calculate a parts data
            rows = imageSize(1) / partSize;
            cols = imageSize(2) / partSize;
            numOfParts = rows * cols;
            partsCorrectOrder = 1:numOfParts;

            % Creating parts array that contain the puzzle parts
            newColorSchemePartsArray =  zeros(partSize, partSize, 3, numOfParts);
            rgbPartsArray =  zeros(partSize, partSize, 3, numOfParts);

            % Splits the image into parts.
            cutImageToParts();
            
            % Initialize parts compatibility, unlike when solving the jigsaw
            % we will fill the entire partsCompVal matrix.
            partsCompVal = zeros(numOfParts,numOfParts,4);
            initializePartsCompatibility();
            initializeConfidence();
            
            initBestNeighbors();
            selectBestSeedCandidate();
            
            bestBuddiesForAllImages(2 *imageIndex - 1, :) = bestBuddies(1, :);
            bestBuddiesForAllImages(2 *imageIndex, :) = bestBuddies(2, :);
        end
    end

    % Contains all unit tests for the nested functions
    function mainCodeTester()
        fprintf('Starting the JigsawSolver main code tester\n');
        fprintf('=======================================\n\n');
        
        % RELAXATION LABELING TESTS
        % %%%%%%%%%%%%%%%%%%%
        
        % sameVector
        test(sameVector([1,2,3],[1,2,3]), 'sameVector - same vectors');
        test(sameVector([1,2,3],[1,2,4]) == 0, 'sameVector - not the same vectors');
        test(sameVector([1,2,3],[1,2,3,4]) == 0, 'sameVector - different length vectors');
        
        % findCommon
        [a,b,c] = findCommon(1,2,1,3);
        test(sameVector([1,2,3], [a,b,c]), 'findCommon - f1 s1 are the same');
        [a,b,c] = findCommon(1,4,4,5);
        test(sameVector([4,1,5], [a,b,c]), 'findCommon - f2 s1 are the same');
        
        % relationToCommon
        test(relationToCommon(1,3,2) == 2, 'relationToCommon - common is the smallest');
        test(relationToCommon(3,1,1) == 2, 'relationToCommon - common is the bigger (left)');
        test(relationToCommon(3,1,2) == 1, 'relationToCommon - common is the bigger (right)');
        test(relationToCommon(3,1,3) == 4, 'relationToCommon - common is the bigger (up)');
        test(relationToCommon(3,1,4) == 3, 'relationToCommon - common is the bigger (down)');
        
        % compatibility
        test(compatibility(1, 2, 3, 5, 1, 1) == 0, 'compatibility - Unrealted connections same realtion');
        test(compatibility(1, 2, 3, 5, 1, 1) == 0, 'compatibility - Unrealted connections different realtion');
        test(compatibility(1, 2, 1, 2, 1, 1) == 0, 'compatibility - Same connection with same relation');
        test(compatibility(1, 2, 1, 2, 1, 3) == -1, 'compatibility - Same connection with different relation');
        test(compatibility(1, 2, 2, 4, 5, 5) == 0, 'compatibility - Common part with both relations as no_realtion');
        test(compatibility(6, 9, 2, 9, 5, 2) == 0, 'compatibility - Common part with one relation as no_realtion');
        test(compatibility(1, 2, 1, 3, 1, 1) == -1, 'compatibility - Common part with the same label meaning');
        test(compatibility(1, 2, 1, 3, 1, 2) == 0, 'compatibility - Common part with the opposite label meaning');
        %test(compatibility(1, 2, 1, 3, 4, 2) == 1, 'compatibility - Common part with the near label meaning');%TODO
        %with same relation');    
        
        % GREEDY ALGORITHM  TESTS
        % %%%%%%%%%%%%%%%%%
        
        % removePartFromUnplaced
        unplacedParts = [1,2,3,4,5];
        unplacedCounter = 5;
        removePartFromUnplaced(5);
        test(unplacedCounter == 4, 'removePartFromUnplaced - Counter was not decreased');
        test(sameVector(unplacedParts, [1,2,3,4,5]), 'removePartFromUnplaced - unplacedParts shold have stayed the same');
        unplacedParts = [1,2,3,4,5];
        unplacedCounter = 4;
        removePartFromUnplaced(1);
        test(unplacedCounter == 3, 'removePartFromUnplaced - Counter was not decreased');
        test(sameVector(unplacedParts, [4,2,3,1,5]), 'removePartFromUnplaced - unplacedParts shold have stayed the same');
        
        % removeCellFromNeighborsVec
        numOfParts = 4;
        neighborVec = [1,2,3,0;1,2,3,0;1,3,2,0];
        removeCellFromNeighborsVec(2,2);
        sameMat = sameVector(neighborVec(1,:),[1,0,3,0]) && sameVector(neighborVec(2,:),[1,0,3,0]) && sameVector(neighborVec(3,:),[1,0,2,0]);
        test(sameMat,'sortNeighborsByKnownParts - did not remove the second neighbor properly');
        removeCellFromNeighborsVec(1,1);
        removeCellFromNeighborsVec(3,3);
        sameMat = sameVector(neighborVec(1,:),[0,0,0,0]) && sameVector(neighborVec(2,:),[0,0,0,0]) && sameVector(neighborVec(3,:),[0,0,0,0]);
        test(sameMat,'sortNeighborsByKnownParts - did remove all neighbors');
        
        % sortNeighborsByKnownParts
        neighborVec = [1,2,3,0;1,2,3,0;1,3,2,0];
        sortNeighborsByKnownParts();
        sameMat = sameVector(neighborVec(1,:),[2,3,1,0]) && sameVector(neighborVec(2,:),[2,3,1,0]) && sameVector(neighborVec(3,:),[3,2,1,0]);
        test(sameMat,'sortNeighborsByKnownParts - did not sort properly');
        
        fprintf('Successfully passed all: %d unit test!\n', testsPassed);
        
        % calcBestBuddiesForAllImages();
        
        % Checking part compatibility functions
        testPartsCompatibilityFunctions();
    end
end