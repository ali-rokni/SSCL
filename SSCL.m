function [all_acc, preci, reca, f1a] = SSCL(source, subs, trainIds, SCCLIds, testIds, movements, exampleSize, allLocs)
rng(1234);
preci = [];
reca = [];
f1a = [];
precision = @(confusionMat) diag(confusionMat)./sum(confusionMat)';
recall = @(confusionMat) diag(confusionMat)./sum(confusionMat,2);
f1Scores = @(confusionMat) 2*(precision(confusionMat).*recall(confusionMat))./(precision(confusionMat)+recall(confusionMat));
meanF1 = @(confusionMat) mean(f1Scores(confusionMat));
rand_tr = 100;
all_acc = [];
for mm=1:rand_tr
    A = randperm(exampleSize);
    for sub2 = subs
        allX = [];
        allY = [];
        allZ = [];
        sub1 = setdiff(subs, sub2);
        locs = setdiff(allLocs, source);
        loc_size = size(locs, 2);
        [X, Y, Z] = readData(movements, sub1 , source, A(trainIds));
        sourceMDL = TreeBagger(3, X, Y, 'Method', 'Classification');
        Map = containers.Map({source(1)}, {sourceMDL});
        loss = [];
        for loc=locs 
           [X, Y] = readData(movements, sub1 , loc, A(trainIds));
            mdl = TreeBagger(3, X, Y, 'Method', 'Classification');
            Map1 = containers.Map({loc},{mdl});
            Map = [Map;Map1];
        end;
        [sourceX, sourceY]= readData(movements, sub2, source, A(SCCLIds));
        labels = predict(sourceMDL, sourceX);
        labels = str2double(labels);
        sum(sourceY == labels) / size(sourceY, 1);
        [X,Y, Z] = readData(movements, sub2 ,locs, A(SCCLIds));
        allX = [allX;X];
        allY = [allY;Y];
        allZ = [allZ;Z];

        Y = [];
        i = 1;

        XperLoc = containers.Map({source(1)}, {[0 0]});
        XLocs = [];
        YLocs = [];
        for loc=locs
            XperLoc(loc) = [];
        end;
        k = 1;
        sourceXs = [];
        sourceLabels = [];
        [aa numLoc] = size(locs);
        for i=1:size(allX)
            minD = 1000000;
            minLs = [];
             sourceXs = [sourceXs; allX(i,:)];
             labels2 = 0;
            for loc=locs
                MDL = Map(loc);
                [labels1, scores1] = predict(MDL, allX(i,:));
                [labels2, scores2] = predict(sourceMDL, sourceX(k, :));
                [labels1, labels2, allY(i,:)];
                D = pdist2(scores1, scores2);
               % D = abs(labels1 - labels2);
    %            if str2double(labels1) == str2double(labels2)
    %                minLs = [minLs loc];
    %            end
                if D < minD
                    minD = D;
                    minLs = [loc];  
                else if D == minD
                        minLs = [minLs loc];
                    end;
                end;
            end;
            sourceLabels = [sourceLabels; labels2];

            if( mod(i, numLoc) == 0)
                k = k + 1;
            end;
        %     minL = minLs(1);
        %     [a b] = size(minLs);
        %     
        %     if b > 1
        %         minD = 100000;
        %         for location = minLs
        %             [c d]= size(XperLoc(location)); 
        %             if c > 0
        %                 [idx, D] = knnsearch(XperLoc(location), allX(i,:));
        %                 
        %                 if D < minD
        %                     minD = D;
        %                     minL = location;
        %                 end;
        %             end;
        %         end;
        %     end;

            for minL= minLs
                XLocs = [XLocs; allX(i,:)];
                YLocs = [YLocs; minL];
                %XperLoc(minL) = [XperLoc(minL); allX(i,:)];
            end;

          %  Y = [Y; minL allZ(i)];
        end;
        % Ys = [];
        % Xs = [];
        % keyset = keys(XperLoc);

        % for key= keyset
        %     items = XperLoc(key{1,1});
        %     [a b] = size(items);
        %     i = 1;
        %     while i <= a && b > 10
        %         Xs = [Xs;items(i, :)];
        %         Ys = [Ys; key{1,1}];
        %         i = i + 1;
        %     end;
        % end;



        mag_range = 2:size(XLocs, 2);
        mdl2 = TreeBagger(3, XLocs(:, mag_range), YLocs, 'Method', 'Classification');
       [X Y Z] = readData(movements, sub2,locs, A(testIds));
        loclabs = str2double(predict(mdl2, X(:, mag_range)));
        acloc = sum(loclabs == Z) / size(Z,1);
        [a b] = size(X);
        sourceLabels = str2double(sourceLabels);
        sys_MDL = TreeBagger(1, sourceXs, sourceLabels, 'Method', 'Classification');
        sys_lab = predict(sys_MDL, X);
        sys_lab = str2double(sys_lab);
        sys_acc = sum((Y-sys_lab) == 0)/ size(X,1);


        pred1 = [];
        pred2 = [];
        pred3 = [];
        pred4 = [];
        pred5 = [];
        pred6 = [];
        i = 1;
        no_rands = 2;
        rands = randi([1,loc_size], no_rands, a);

        while i < a+1
            labels = predict(mdl2, X(i,mag_range));
            labels = str2double(labels);
            mdl11 = Map(labels);
            mdl12 = Map(Z(i));
            pred1 = [pred1; str2double(predict(mdl11, X(i,:)))];
            pred2 = [pred2; str2double(predict(mdl12, X(i,:)))];
            randres = [];
            for j=1:no_rands
                mdl13 = Map(locs(rands(j, i)));
                randres = [randres str2double(predict(mdl13, X(i,:)))];
            end;
            pred3 = [pred3; randres];
            pred4 = [pred4; str2double(predict(sourceMDL, X(i,:)))];
            temp = [];
            for j=1:loc_size
                mdlj = Map(locs(j));
                temp = [temp; str2double(predict(mdlj, X(i,:)))];
            end;
            pred5 = [pred5; mode(temp)];
            pred6 = [pred6; temp'];
            i = i + 1;
        end;

        acc = sum((Y-pred1) == 0)/ a;
        [Y pred1];
        upp_acc = sum((Y-pred2) == 0)/ a;
        naive_acc = sum((Y-pred4) == 0)/ a;

        vote = sum((Y-pred5) == 0)/ a;
        res = 0;

        for i=1:no_rands
             res = res + sum((Y-pred3(:,i) == 0));
        end;
        rand_acc =res / (a*no_rands);
        accs = [upp_acc, acc, sys_acc, naive_acc, vote, rand_acc, acloc];
        all_acc = [all_acc; accs];

        pred3 = reshape(pred3, [],1);
        YN = repmat(Y, no_rands,1);
        cm1 = confusionmat(Y,pred2);
        cm2 = confusionmat(Y,pred1);
        cm3 = confusionmat(Y, sys_lab);
        cm4 = confusionmat(Y,pred4);
        cm5 = confusionmat(Y, pred5);
        cm6 = confusionmat(YN,pred3);
        preci_t = [precision(cm1) precision(cm2) precision(cm3) precision(cm4) precision(cm5) precision(cm6)];
        preci_t(isnan(preci_t))=0;
        preci = [preci; mean(preci_t)];
        reca_t =[recall(cm1) recall(cm2) recall(cm3) recall(cm4) recall(cm5) recall(cm6)];
        reca_t(isnan(reca_t))=0;
        reca = [reca; mean(reca_t)];

        f1_t = 2 * mean(preci_t) .* mean(reca_t) ./ (mean(preci_t) + mean(reca_t));
        f1a = [f1a;f1_t];
    end

end
