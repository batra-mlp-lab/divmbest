function parts_org = suppress_mode(parts_org,parts_msg,root_x,root_y,root_t,lambda,type)
% follow backtracking ptrs and modify unary potentials
parts_org(1).score(root_y,root_x,root_t) = ...
    parts_org(1).score(root_y,root_x,root_t) + lambda;
ptr(1,:) = [root_x root_y root_t];
for i = 2:length(parts_msg)
    parent_i = parts_msg(i).parent;
    x = ptr(parent_i,1);
    y = ptr(parent_i,2);
    t = ptr(parent_i,3);
    
    ptr(i,1) = parts_msg(i).Ix(y,x,t);
    ptr(i,2) = parts_msg(i).Iy(y,x,t);
    ptr(i,3) = parts_msg(i).Ik(y,x,t);
    x = ptr(i,1);
    y = ptr(i,2);
    t = ptr(i,3);
    
    if(strcmp(type, 'perturb'))
        U = rand;
        gumbel = log(-log(U));
        parts_org(i).score(y,x,t) = ...
            parts_org(i).score(y,x,t) + lambda*gumbel;
    end
    
    if(strcmp(type, 'divmbest'))
        parts_org(i).score(y,x,t) = ...
            parts_org(i).score(y,x,t) + lambda;
    end
end
end
