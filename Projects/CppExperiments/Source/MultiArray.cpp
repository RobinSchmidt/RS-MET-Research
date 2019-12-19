



/*
Maybe, we make the storage layout user adjustable? Modifications that would be necessarry in order
to let the first index have a stride of 1 instead of the last (resulting in column-major storage in
case of 2D arrays): 

the loop in updateStrides would have to become:
int i = rank-1; int s = 1; while(i >= 0) { strides[i] = s; s *= shape[rank-i-1]; --i; }

the ()-operator would become:
return data[ flatIndex((int)shape.size()-1, i, rest...) ];

index computaion in flatIndex(int depth, int i, Rest... rest) would become:
return flatIndex(depth, i) + flatIndex(depth-1, rest...);

*/