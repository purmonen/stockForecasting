classdef MultilayerPerceptron < handle
    properties
        w1 = [];
        w2 = [];
        err = [];
        validationErr = [];
        patterns = [];
        targets = [];
        validationTargets = [];
        validationPatterns = [];
        hiddenNodes = 5;
        plottingEnabled = true;
        iterations = 1000;
        plotFrequency = 1;
        alpha = 0.9;
        eta = 0.1;
        b = 1;
    end
    methods
        function obj = ANNPerceptron()
        end
        function out = phi(obj, hin)
            out = 2 ./ (1+exp(-obj.b*hin)) - 1;
            
        end
        function out = dphi(obj, hin)
            %out = (1+obj.phi(hin)).*(1-obj.phi(hin)) ./ 2;
            out = (2*obj.b * exp(-obj.b*hin))./(1 + exp(-obj.b*hin)).^2;
        end
        function out = recall(obj, patterns)
            hin = obj.w1 * [patterns; ones(1,size(patterns,2))];
            hout = [obj.phi(hin); ones(1, size(hin,2))];
            oin = obj.w2 * hout;
            out = obj.phi(oin);
        end
        function out = recallHidden(obj, patterns)
            hin = obj.w1 * [patterns; ones(1,size(patterns,2))];
            hout = [obj.phi(hin); ones(1, size(hin,2))];
            out = hout(1:end-1, :);
        end
        function train(obj, patterns, targets)
            obj.patterns = patterns;
            obj.targets = targets;
            
            patterns = [patterns; ones(1,size(patterns,2))];
            permute = randperm(size(targets,2));
            patterns = patterns(:, permute);
            targets = targets(:, permute);
            
            % Multi layer perceptron
            obj.w1 = randn(obj.hiddenNodes, size(patterns,1));
            obj.w2 = randn(size(targets, 1), size(obj.w1,1)+1);
            
            dw1 = obj.w1*0;
            dw2 = obj.w2*0;
            for epoch = 1:obj.iterations
                %obj.trainingCallback();
                if obj.plottingEnabled && mod(epoch, obj.plotFrequency) == 0
                    obj.plotStuff()
                end
                
                % Forward pass
                pat = patterns;
                hin = obj.w1 * patterns;
                hout = [obj.phi(hin); ones(1, size(hin,2))];
                %hout = sign(hout);
                oin = obj.w2 * hout;
                out = obj.phi(oin);
                
                % Backward pass
                %deltao = (out - targets) .* ((1 + out) .* (1 - out)) * 0.5;
                %deltah = (obj.w2' * deltao) .* ((1 + hout) .* (1 - hout)) * 0.5;
                deltao = (out - targets) .* obj.dphi(out);
                deltah = (obj.w2' * deltao) .* obj.dphi(hout);
                deltah = deltah(1:end-1, :);
                
                % Updating weights
                dw1 = (dw1 .* obj.alpha) - (deltah * pat') .* (1-obj.alpha);
                dw2 = (dw2 .* obj.alpha) - (deltao * hout') .* (1-obj.alpha);
                obj.w1 = obj.w1 + dw1 .* obj.eta;
                obj.w2 = obj.w2 + dw2 .* obj.eta;
                
                obj.err = [obj.err, [sum(sum(abs(sign(obj.recall(obj.patterns)) - obj.targets) .* 0.5)); sum(sum(abs(obj.recall(obj.patterns) - obj.targets)))]];
                if (size(obj.validationPatterns,1) ~= 0)
                    obj.validationErr = [obj.validationErr, [sum(sum(abs(sign(obj.recall(obj.validationPatterns)) - obj.validationTargets) .* 0.5)); sum(sum(abs(obj.recall(obj.validationPatterns) - obj.validationTargets)))]];
                end
                
            end
        end
        function plotStuff(obj)
            grid = [];
            points = 50;
            x=linspace(min(floor(obj.patterns(1,:))), ceil(max(obj.patterns(1,:))), points);
            y=linspace(min(floor(obj.patterns(2,:))), ceil(max(obj.patterns(2,:))), points);
            for i=x;
                for j=y;
                    grid = [grid; [i, j]];
                end
            end
            grid = grid';
            z = ones(size(obj.patterns,1) - size(grid,1), size(grid,2));
            grid = [grid; z];
            if (size(grid, 1) == size(obj.patterns, 1))
                out = obj.recall(grid);
                if (size(out,1) == 1)
                    %                     subplot(3,1,1)
                    %                     plot(grid(1, out >= 0),grid(2, out >= 0),'.b',...
                    %                         grid(1, out < 0),grid(2, out < 0),'+r',...
                    %                         obj.patterns(1, obj.targets >= 0),obj.patterns(2, obj.targets >= 0),'*g',...
                    %                         obj.patterns(1, obj.targets < 0),obj.patterns(2, obj.targets < 0),'*y');
                    
                    %zzz = reshape(obj.targets, 50, 50);
                    %mesh (x, y, z, 'FaceAlph', 0.5, 'FaceColor','interp','FaceLighting','gouraud');
                    hold on
                    
                    zz = reshape(out, 50, 50);
                    %subplot(3,1,2)
                    %mesh(x,y,zz);
                    %axis([-5 5 -5 5 -0.7 0.7]);
                    drawnow;
                    hold off
                end
                
            end
%             subplot(2,1,1)
%             plot(1:size(obj.err,2), [obj.err])
%             xlabel('Iterations')
%             ylabel('Errors')
%             legend('Classification errors', 'Network error')
%             
%             subplot(2,1,2)
%             plot(1:size(obj.validationErr,2), obj.validationErr)
%             xlabel('Iterations')
%             ylabel('Validation Errors')
%             legend('Validation classification errors', 'Validation network error')
%             drawnow;
        end
        function plotErrors(obj)
            subplot(2,2,1)
            plot(1:size(obj.err,2), obj.err(2,:))
            title('Training data')
            xlabel('Iterations')
            ylabel('Errors')
            
            subplot(2,2,2)
            plot(1:size(obj.validationErr,2), obj.validationErr(2,:))
            title('Validation data')
            xlabel('Iterations')
            ylabel('Errors')
            drawnow;
        end
    end 
end