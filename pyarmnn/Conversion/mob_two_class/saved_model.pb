??)
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

?
DepthwiseConv2dNative

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

?
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%??8"&
exponential_avg_factorfloat%  ??";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
?
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
?
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu6
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.6.02unknown8ĸ(
l
save_counterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_namesave_counter
e
 save_counter/Read/ReadVariableOpReadVariableOpsave_counter*
_output_shapes
: *
dtype0	
?
8MobilenetV1/Conv2d_1_depthwise/BatchNorm/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*I
shared_name:8MobilenetV1/Conv2d_1_depthwise/BatchNorm/moving_variance
?
LMobilenetV1/Conv2d_1_depthwise/BatchNorm/moving_variance/Read/ReadVariableOpReadVariableOp8MobilenetV1/Conv2d_1_depthwise/BatchNorm/moving_variance*
_output_shapes
:*
dtype0
?
.MobilenetV1/Conv2d_3_depthwise/BatchNorm/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*?
shared_name0.MobilenetV1/Conv2d_3_depthwise/BatchNorm/gamma
?
BMobilenetV1/Conv2d_3_depthwise/BatchNorm/gamma/Read/ReadVariableOpReadVariableOp.MobilenetV1/Conv2d_3_depthwise/BatchNorm/gamma*
_output_shapes
:`*
dtype0
?
.MobilenetV1/Conv2d_7_pointwise/BatchNorm/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*?
shared_name0.MobilenetV1/Conv2d_7_pointwise/BatchNorm/gamma
?
BMobilenetV1/Conv2d_7_pointwise/BatchNorm/gamma/Read/ReadVariableOpReadVariableOp.MobilenetV1/Conv2d_7_pointwise/BatchNorm/gamma*
_output_shapes	
:?*
dtype0
?
-MobilenetV1/Conv2d_8_pointwise/BatchNorm/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*>
shared_name/-MobilenetV1/Conv2d_8_pointwise/BatchNorm/beta
?
AMobilenetV1/Conv2d_8_pointwise/BatchNorm/beta/Read/ReadVariableOpReadVariableOp-MobilenetV1/Conv2d_8_pointwise/BatchNorm/beta*
_output_shapes	
:?*
dtype0
?
.MobilenetV1/Conv2d_11_pointwise/BatchNorm/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*?
shared_name0.MobilenetV1/Conv2d_11_pointwise/BatchNorm/beta
?
BMobilenetV1/Conv2d_11_pointwise/BatchNorm/beta/Read/ReadVariableOpReadVariableOp.MobilenetV1/Conv2d_11_pointwise/BatchNorm/beta*
_output_shapes	
:?*
dtype0
?
0MobilenetV1/Conv2d_4_depthwise/depthwise_weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*A
shared_name20MobilenetV1/Conv2d_4_depthwise/depthwise_weights
?
DMobilenetV1/Conv2d_4_depthwise/depthwise_weights/Read/ReadVariableOpReadVariableOp0MobilenetV1/Conv2d_4_depthwise/depthwise_weights*&
_output_shapes
:`*
dtype0
?
(MobilenetV1/Logits/Conv2d_1c_1x1/weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*9
shared_name*(MobilenetV1/Logits/Conv2d_1c_1x1/weights
?
<MobilenetV1/Logits/Conv2d_1c_1x1/weights/Read/ReadVariableOpReadVariableOp(MobilenetV1/Logits/Conv2d_1c_1x1/weights*(
_output_shapes
:??*
dtype0
?
*MobilenetV1/Conv2d_0/BatchNorm/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*MobilenetV1/Conv2d_0/BatchNorm/moving_mean
?
>MobilenetV1/Conv2d_0/BatchNorm/moving_mean/Read/ReadVariableOpReadVariableOp*MobilenetV1/Conv2d_0/BatchNorm/moving_mean*
_output_shapes
:*
dtype0
?
4MobilenetV1/Conv2d_3_pointwise/BatchNorm/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*E
shared_name64MobilenetV1/Conv2d_3_pointwise/BatchNorm/moving_mean
?
HMobilenetV1/Conv2d_3_pointwise/BatchNorm/moving_mean/Read/ReadVariableOpReadVariableOp4MobilenetV1/Conv2d_3_pointwise/BatchNorm/moving_mean*
_output_shapes
:`*
dtype0
?
4MobilenetV1/Conv2d_4_depthwise/BatchNorm/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*E
shared_name64MobilenetV1/Conv2d_4_depthwise/BatchNorm/moving_mean
?
HMobilenetV1/Conv2d_4_depthwise/BatchNorm/moving_mean/Read/ReadVariableOpReadVariableOp4MobilenetV1/Conv2d_4_depthwise/BatchNorm/moving_mean*
_output_shapes
:`*
dtype0
?
8MobilenetV1/Conv2d_5_depthwise/BatchNorm/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*I
shared_name:8MobilenetV1/Conv2d_5_depthwise/BatchNorm/moving_variance
?
LMobilenetV1/Conv2d_5_depthwise/BatchNorm/moving_variance/Read/ReadVariableOpReadVariableOp8MobilenetV1/Conv2d_5_depthwise/BatchNorm/moving_variance*
_output_shapes	
:?*
dtype0
?
.MobilenetV1/Conv2d_6_depthwise/BatchNorm/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*?
shared_name0.MobilenetV1/Conv2d_6_depthwise/BatchNorm/gamma
?
BMobilenetV1/Conv2d_6_depthwise/BatchNorm/gamma/Read/ReadVariableOpReadVariableOp.MobilenetV1/Conv2d_6_depthwise/BatchNorm/gamma*
_output_shapes	
:?*
dtype0
?
8MobilenetV1/Conv2d_7_pointwise/BatchNorm/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*I
shared_name:8MobilenetV1/Conv2d_7_pointwise/BatchNorm/moving_variance
?
LMobilenetV1/Conv2d_7_pointwise/BatchNorm/moving_variance/Read/ReadVariableOpReadVariableOp8MobilenetV1/Conv2d_7_pointwise/BatchNorm/moving_variance*
_output_shapes	
:?*
dtype0
?
8MobilenetV1/Conv2d_9_depthwise/BatchNorm/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*I
shared_name:8MobilenetV1/Conv2d_9_depthwise/BatchNorm/moving_variance
?
LMobilenetV1/Conv2d_9_depthwise/BatchNorm/moving_variance/Read/ReadVariableOpReadVariableOp8MobilenetV1/Conv2d_9_depthwise/BatchNorm/moving_variance*
_output_shapes	
:?*
dtype0
?
8MobilenetV1/Conv2d_9_pointwise/BatchNorm/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*I
shared_name:8MobilenetV1/Conv2d_9_pointwise/BatchNorm/moving_variance
?
LMobilenetV1/Conv2d_9_pointwise/BatchNorm/moving_variance/Read/ReadVariableOpReadVariableOp8MobilenetV1/Conv2d_9_pointwise/BatchNorm/moving_variance*
_output_shapes	
:?*
dtype0
?
'MobilenetV1/Conv2d_11_pointwise/weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*8
shared_name)'MobilenetV1/Conv2d_11_pointwise/weights
?
;MobilenetV1/Conv2d_11_pointwise/weights/Read/ReadVariableOpReadVariableOp'MobilenetV1/Conv2d_11_pointwise/weights*(
_output_shapes
:??*
dtype0
?
4MobilenetV1/Conv2d_1_pointwise/BatchNorm/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*E
shared_name64MobilenetV1/Conv2d_1_pointwise/BatchNorm/moving_mean
?
HMobilenetV1/Conv2d_1_pointwise/BatchNorm/moving_mean/Read/ReadVariableOpReadVariableOp4MobilenetV1/Conv2d_1_pointwise/BatchNorm/moving_mean*
_output_shapes
:0*
dtype0
?
4MobilenetV1/Conv2d_5_depthwise/BatchNorm/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*E
shared_name64MobilenetV1/Conv2d_5_depthwise/BatchNorm/moving_mean
?
HMobilenetV1/Conv2d_5_depthwise/BatchNorm/moving_mean/Read/ReadVariableOpReadVariableOp4MobilenetV1/Conv2d_5_depthwise/BatchNorm/moving_mean*
_output_shapes	
:?*
dtype0
?
5MobilenetV1/Conv2d_10_pointwise/BatchNorm/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*F
shared_name75MobilenetV1/Conv2d_10_pointwise/BatchNorm/moving_mean
?
IMobilenetV1/Conv2d_10_pointwise/BatchNorm/moving_mean/Read/ReadVariableOpReadVariableOp5MobilenetV1/Conv2d_10_pointwise/BatchNorm/moving_mean*
_output_shapes	
:?*
dtype0
?
-MobilenetV1/Conv2d_7_pointwise/BatchNorm/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*>
shared_name/-MobilenetV1/Conv2d_7_pointwise/BatchNorm/beta
?
AMobilenetV1/Conv2d_7_pointwise/BatchNorm/beta/Read/ReadVariableOpReadVariableOp-MobilenetV1/Conv2d_7_pointwise/BatchNorm/beta*
_output_shapes	
:?*
dtype0
?
MobilenetV1/Conv2d_0/weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameMobilenetV1/Conv2d_0/weights
?
0MobilenetV1/Conv2d_0/weights/Read/ReadVariableOpReadVariableOpMobilenetV1/Conv2d_0/weights*&
_output_shapes
:*
dtype0
?
8MobilenetV1/Conv2d_2_depthwise/BatchNorm/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*I
shared_name:8MobilenetV1/Conv2d_2_depthwise/BatchNorm/moving_variance
?
LMobilenetV1/Conv2d_2_depthwise/BatchNorm/moving_variance/Read/ReadVariableOpReadVariableOp8MobilenetV1/Conv2d_2_depthwise/BatchNorm/moving_variance*
_output_shapes
:0*
dtype0
?
8MobilenetV1/Conv2d_4_depthwise/BatchNorm/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*I
shared_name:8MobilenetV1/Conv2d_4_depthwise/BatchNorm/moving_variance
?
LMobilenetV1/Conv2d_4_depthwise/BatchNorm/moving_variance/Read/ReadVariableOpReadVariableOp8MobilenetV1/Conv2d_4_depthwise/BatchNorm/moving_variance*
_output_shapes
:`*
dtype0
?
4MobilenetV1/Conv2d_6_pointwise/BatchNorm/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*E
shared_name64MobilenetV1/Conv2d_6_pointwise/BatchNorm/moving_mean
?
HMobilenetV1/Conv2d_6_pointwise/BatchNorm/moving_mean/Read/ReadVariableOpReadVariableOp4MobilenetV1/Conv2d_6_pointwise/BatchNorm/moving_mean*
_output_shapes	
:?*
dtype0
?
-MobilenetV1/Conv2d_9_depthwise/BatchNorm/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*>
shared_name/-MobilenetV1/Conv2d_9_depthwise/BatchNorm/beta
?
AMobilenetV1/Conv2d_9_depthwise/BatchNorm/beta/Read/ReadVariableOpReadVariableOp-MobilenetV1/Conv2d_9_depthwise/BatchNorm/beta*
_output_shapes	
:?*
dtype0
?
/MobilenetV1/Conv2d_11_depthwise/BatchNorm/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*@
shared_name1/MobilenetV1/Conv2d_11_depthwise/BatchNorm/gamma
?
CMobilenetV1/Conv2d_11_depthwise/BatchNorm/gamma/Read/ReadVariableOpReadVariableOp/MobilenetV1/Conv2d_11_depthwise/BatchNorm/gamma*
_output_shapes	
:?*
dtype0
?
&MobilenetV1/Conv2d_6_pointwise/weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*7
shared_name(&MobilenetV1/Conv2d_6_pointwise/weights
?
:MobilenetV1/Conv2d_6_pointwise/weights/Read/ReadVariableOpReadVariableOp&MobilenetV1/Conv2d_6_pointwise/weights*(
_output_shapes
:??*
dtype0
?
-MobilenetV1/Conv2d_6_pointwise/BatchNorm/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*>
shared_name/-MobilenetV1/Conv2d_6_pointwise/BatchNorm/beta
?
AMobilenetV1/Conv2d_6_pointwise/BatchNorm/beta/Read/ReadVariableOpReadVariableOp-MobilenetV1/Conv2d_6_pointwise/BatchNorm/beta*
_output_shapes	
:?*
dtype0
?
-MobilenetV1/Conv2d_1_depthwise/BatchNorm/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*>
shared_name/-MobilenetV1/Conv2d_1_depthwise/BatchNorm/beta
?
AMobilenetV1/Conv2d_1_depthwise/BatchNorm/beta/Read/ReadVariableOpReadVariableOp-MobilenetV1/Conv2d_1_depthwise/BatchNorm/beta*
_output_shapes
:*
dtype0
?
.MobilenetV1/Conv2d_3_pointwise/BatchNorm/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*?
shared_name0.MobilenetV1/Conv2d_3_pointwise/BatchNorm/gamma
?
BMobilenetV1/Conv2d_3_pointwise/BatchNorm/gamma/Read/ReadVariableOpReadVariableOp.MobilenetV1/Conv2d_3_pointwise/BatchNorm/gamma*
_output_shapes
:`*
dtype0
?
.MobilenetV1/Conv2d_10_depthwise/BatchNorm/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*?
shared_name0.MobilenetV1/Conv2d_10_depthwise/BatchNorm/beta
?
BMobilenetV1/Conv2d_10_depthwise/BatchNorm/beta/Read/ReadVariableOpReadVariableOp.MobilenetV1/Conv2d_10_depthwise/BatchNorm/beta*
_output_shapes	
:?*
dtype0
?
'MobilenetV1/Logits/Conv2d_1c_1x1/biasesVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*8
shared_name)'MobilenetV1/Logits/Conv2d_1c_1x1/biases
?
;MobilenetV1/Logits/Conv2d_1c_1x1/biases/Read/ReadVariableOpReadVariableOp'MobilenetV1/Logits/Conv2d_1c_1x1/biases*
_output_shapes	
:?*
dtype0
?
4MobilenetV1/Conv2d_8_pointwise/BatchNorm/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*E
shared_name64MobilenetV1/Conv2d_8_pointwise/BatchNorm/moving_mean
?
HMobilenetV1/Conv2d_8_pointwise/BatchNorm/moving_mean/Read/ReadVariableOpReadVariableOp4MobilenetV1/Conv2d_8_pointwise/BatchNorm/moving_mean*
_output_shapes	
:?*
dtype0
?
-MobilenetV1/Conv2d_5_pointwise/BatchNorm/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*>
shared_name/-MobilenetV1/Conv2d_5_pointwise/BatchNorm/beta
?
AMobilenetV1/Conv2d_5_pointwise/BatchNorm/beta/Read/ReadVariableOpReadVariableOp-MobilenetV1/Conv2d_5_pointwise/BatchNorm/beta*
_output_shapes	
:?*
dtype0
?
0MobilenetV1/Conv2d_2_depthwise/depthwise_weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*A
shared_name20MobilenetV1/Conv2d_2_depthwise/depthwise_weights
?
DMobilenetV1/Conv2d_2_depthwise/depthwise_weights/Read/ReadVariableOpReadVariableOp0MobilenetV1/Conv2d_2_depthwise/depthwise_weights*&
_output_shapes
:0*
dtype0
?
&MobilenetV1/Conv2d_1_pointwise/weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*7
shared_name(&MobilenetV1/Conv2d_1_pointwise/weights
?
:MobilenetV1/Conv2d_1_pointwise/weights/Read/ReadVariableOpReadVariableOp&MobilenetV1/Conv2d_1_pointwise/weights*&
_output_shapes
:0*
dtype0
?
.MobilenetV1/Conv2d_1_pointwise/BatchNorm/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*?
shared_name0.MobilenetV1/Conv2d_1_pointwise/BatchNorm/gamma
?
BMobilenetV1/Conv2d_1_pointwise/BatchNorm/gamma/Read/ReadVariableOpReadVariableOp.MobilenetV1/Conv2d_1_pointwise/BatchNorm/gamma*
_output_shapes
:0*
dtype0
?
-MobilenetV1/Conv2d_5_depthwise/BatchNorm/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*>
shared_name/-MobilenetV1/Conv2d_5_depthwise/BatchNorm/beta
?
AMobilenetV1/Conv2d_5_depthwise/BatchNorm/beta/Read/ReadVariableOpReadVariableOp-MobilenetV1/Conv2d_5_depthwise/BatchNorm/beta*
_output_shapes	
:?*
dtype0
?
8MobilenetV1/Conv2d_5_pointwise/BatchNorm/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*I
shared_name:8MobilenetV1/Conv2d_5_pointwise/BatchNorm/moving_variance
?
LMobilenetV1/Conv2d_5_pointwise/BatchNorm/moving_variance/Read/ReadVariableOpReadVariableOp8MobilenetV1/Conv2d_5_pointwise/BatchNorm/moving_variance*
_output_shapes	
:?*
dtype0
?
.MobilenetV1/Conv2d_8_depthwise/BatchNorm/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*?
shared_name0.MobilenetV1/Conv2d_8_depthwise/BatchNorm/gamma
?
BMobilenetV1/Conv2d_8_depthwise/BatchNorm/gamma/Read/ReadVariableOpReadVariableOp.MobilenetV1/Conv2d_8_depthwise/BatchNorm/gamma*
_output_shapes	
:?*
dtype0
?
1MobilenetV1/Conv2d_13_depthwise/depthwise_weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*B
shared_name31MobilenetV1/Conv2d_13_depthwise/depthwise_weights
?
EMobilenetV1/Conv2d_13_depthwise/depthwise_weights/Read/ReadVariableOpReadVariableOp1MobilenetV1/Conv2d_13_depthwise/depthwise_weights*'
_output_shapes
:?*
dtype0
?
8MobilenetV1/Conv2d_6_depthwise/BatchNorm/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*I
shared_name:8MobilenetV1/Conv2d_6_depthwise/BatchNorm/moving_variance
?
LMobilenetV1/Conv2d_6_depthwise/BatchNorm/moving_variance/Read/ReadVariableOpReadVariableOp8MobilenetV1/Conv2d_6_depthwise/BatchNorm/moving_variance*
_output_shapes	
:?*
dtype0
?
&MobilenetV1/Conv2d_9_pointwise/weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*7
shared_name(&MobilenetV1/Conv2d_9_pointwise/weights
?
:MobilenetV1/Conv2d_9_pointwise/weights/Read/ReadVariableOpReadVariableOp&MobilenetV1/Conv2d_9_pointwise/weights*(
_output_shapes
:??*
dtype0
?
9MobilenetV1/Conv2d_10_depthwise/BatchNorm/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*J
shared_name;9MobilenetV1/Conv2d_10_depthwise/BatchNorm/moving_variance
?
MMobilenetV1/Conv2d_10_depthwise/BatchNorm/moving_variance/Read/ReadVariableOpReadVariableOp9MobilenetV1/Conv2d_10_depthwise/BatchNorm/moving_variance*
_output_shapes	
:?*
dtype0
?
.MobilenetV1/Conv2d_12_pointwise/BatchNorm/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*?
shared_name0.MobilenetV1/Conv2d_12_pointwise/BatchNorm/beta
?
BMobilenetV1/Conv2d_12_pointwise/BatchNorm/beta/Read/ReadVariableOpReadVariableOp.MobilenetV1/Conv2d_12_pointwise/BatchNorm/beta*
_output_shapes	
:?*
dtype0
?
-MobilenetV1/Conv2d_2_depthwise/BatchNorm/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*>
shared_name/-MobilenetV1/Conv2d_2_depthwise/BatchNorm/beta
?
AMobilenetV1/Conv2d_2_depthwise/BatchNorm/beta/Read/ReadVariableOpReadVariableOp-MobilenetV1/Conv2d_2_depthwise/BatchNorm/beta*
_output_shapes
:0*
dtype0
?
4MobilenetV1/Conv2d_2_pointwise/BatchNorm/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*E
shared_name64MobilenetV1/Conv2d_2_pointwise/BatchNorm/moving_mean
?
HMobilenetV1/Conv2d_2_pointwise/BatchNorm/moving_mean/Read/ReadVariableOpReadVariableOp4MobilenetV1/Conv2d_2_pointwise/BatchNorm/moving_mean*
_output_shapes
:`*
dtype0
?
-MobilenetV1/Conv2d_3_pointwise/BatchNorm/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*>
shared_name/-MobilenetV1/Conv2d_3_pointwise/BatchNorm/beta
?
AMobilenetV1/Conv2d_3_pointwise/BatchNorm/beta/Read/ReadVariableOpReadVariableOp-MobilenetV1/Conv2d_3_pointwise/BatchNorm/beta*
_output_shapes
:`*
dtype0
?
4MobilenetV1/Conv2d_6_depthwise/BatchNorm/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*E
shared_name64MobilenetV1/Conv2d_6_depthwise/BatchNorm/moving_mean
?
HMobilenetV1/Conv2d_6_depthwise/BatchNorm/moving_mean/Read/ReadVariableOpReadVariableOp4MobilenetV1/Conv2d_6_depthwise/BatchNorm/moving_mean*
_output_shapes	
:?*
dtype0
?
/MobilenetV1/Conv2d_10_pointwise/BatchNorm/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*@
shared_name1/MobilenetV1/Conv2d_10_pointwise/BatchNorm/gamma
?
CMobilenetV1/Conv2d_10_pointwise/BatchNorm/gamma/Read/ReadVariableOpReadVariableOp/MobilenetV1/Conv2d_10_pointwise/BatchNorm/gamma*
_output_shapes	
:?*
dtype0
?
/MobilenetV1/Conv2d_12_depthwise/BatchNorm/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*@
shared_name1/MobilenetV1/Conv2d_12_depthwise/BatchNorm/gamma
?
CMobilenetV1/Conv2d_12_depthwise/BatchNorm/gamma/Read/ReadVariableOpReadVariableOp/MobilenetV1/Conv2d_12_depthwise/BatchNorm/gamma*
_output_shapes	
:?*
dtype0
?
5MobilenetV1/Conv2d_13_depthwise/BatchNorm/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*F
shared_name75MobilenetV1/Conv2d_13_depthwise/BatchNorm/moving_mean
?
IMobilenetV1/Conv2d_13_depthwise/BatchNorm/moving_mean/Read/ReadVariableOpReadVariableOp5MobilenetV1/Conv2d_13_depthwise/BatchNorm/moving_mean*
_output_shapes	
:?*
dtype0
?
0MobilenetV1/Conv2d_5_depthwise/depthwise_weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*A
shared_name20MobilenetV1/Conv2d_5_depthwise/depthwise_weights
?
DMobilenetV1/Conv2d_5_depthwise/depthwise_weights/Read/ReadVariableOpReadVariableOp0MobilenetV1/Conv2d_5_depthwise/depthwise_weights*'
_output_shapes
:?*
dtype0
?
.MobilenetV1/Conv2d_13_depthwise/BatchNorm/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*?
shared_name0.MobilenetV1/Conv2d_13_depthwise/BatchNorm/beta
?
BMobilenetV1/Conv2d_13_depthwise/BatchNorm/beta/Read/ReadVariableOpReadVariableOp.MobilenetV1/Conv2d_13_depthwise/BatchNorm/beta*
_output_shapes	
:?*
dtype0
?
.MobilenetV1/Conv2d_8_pointwise/BatchNorm/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*?
shared_name0.MobilenetV1/Conv2d_8_pointwise/BatchNorm/gamma
?
BMobilenetV1/Conv2d_8_pointwise/BatchNorm/gamma/Read/ReadVariableOpReadVariableOp.MobilenetV1/Conv2d_8_pointwise/BatchNorm/gamma*
_output_shapes	
:?*
dtype0
?
-MobilenetV1/Conv2d_3_depthwise/BatchNorm/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*>
shared_name/-MobilenetV1/Conv2d_3_depthwise/BatchNorm/beta
?
AMobilenetV1/Conv2d_3_depthwise/BatchNorm/beta/Read/ReadVariableOpReadVariableOp-MobilenetV1/Conv2d_3_depthwise/BatchNorm/beta*
_output_shapes
:`*
dtype0
?
.MobilenetV1/Conv2d_7_depthwise/BatchNorm/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*?
shared_name0.MobilenetV1/Conv2d_7_depthwise/BatchNorm/gamma
?
BMobilenetV1/Conv2d_7_depthwise/BatchNorm/gamma/Read/ReadVariableOpReadVariableOp.MobilenetV1/Conv2d_7_depthwise/BatchNorm/gamma*
_output_shapes	
:?*
dtype0
?
8MobilenetV1/Conv2d_8_depthwise/BatchNorm/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*I
shared_name:8MobilenetV1/Conv2d_8_depthwise/BatchNorm/moving_variance
?
LMobilenetV1/Conv2d_8_depthwise/BatchNorm/moving_variance/Read/ReadVariableOpReadVariableOp8MobilenetV1/Conv2d_8_depthwise/BatchNorm/moving_variance*
_output_shapes	
:?*
dtype0
?
5MobilenetV1/Conv2d_10_depthwise/BatchNorm/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*F
shared_name75MobilenetV1/Conv2d_10_depthwise/BatchNorm/moving_mean
?
IMobilenetV1/Conv2d_10_depthwise/BatchNorm/moving_mean/Read/ReadVariableOpReadVariableOp5MobilenetV1/Conv2d_10_depthwise/BatchNorm/moving_mean*
_output_shapes	
:?*
dtype0
?
9MobilenetV1/Conv2d_10_pointwise/BatchNorm/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*J
shared_name;9MobilenetV1/Conv2d_10_pointwise/BatchNorm/moving_variance
?
MMobilenetV1/Conv2d_10_pointwise/BatchNorm/moving_variance/Read/ReadVariableOpReadVariableOp9MobilenetV1/Conv2d_10_pointwise/BatchNorm/moving_variance*
_output_shapes	
:?*
dtype0
?
'MobilenetV1/Conv2d_12_pointwise/weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*8
shared_name)'MobilenetV1/Conv2d_12_pointwise/weights
?
;MobilenetV1/Conv2d_12_pointwise/weights/Read/ReadVariableOpReadVariableOp'MobilenetV1/Conv2d_12_pointwise/weights*(
_output_shapes
:??*
dtype0
?
9MobilenetV1/Conv2d_12_pointwise/BatchNorm/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*J
shared_name;9MobilenetV1/Conv2d_12_pointwise/BatchNorm/moving_variance
?
MMobilenetV1/Conv2d_12_pointwise/BatchNorm/moving_variance/Read/ReadVariableOpReadVariableOp9MobilenetV1/Conv2d_12_pointwise/BatchNorm/moving_variance*
_output_shapes	
:?*
dtype0
?
.MobilenetV1/Conv2d_4_depthwise/BatchNorm/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*?
shared_name0.MobilenetV1/Conv2d_4_depthwise/BatchNorm/gamma
?
BMobilenetV1/Conv2d_4_depthwise/BatchNorm/gamma/Read/ReadVariableOpReadVariableOp.MobilenetV1/Conv2d_4_depthwise/BatchNorm/gamma*
_output_shapes
:`*
dtype0
?
4MobilenetV1/Conv2d_8_depthwise/BatchNorm/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*E
shared_name64MobilenetV1/Conv2d_8_depthwise/BatchNorm/moving_mean
?
HMobilenetV1/Conv2d_8_depthwise/BatchNorm/moving_mean/Read/ReadVariableOpReadVariableOp4MobilenetV1/Conv2d_8_depthwise/BatchNorm/moving_mean*
_output_shapes	
:?*
dtype0
?
5MobilenetV1/Conv2d_13_pointwise/BatchNorm/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*F
shared_name75MobilenetV1/Conv2d_13_pointwise/BatchNorm/moving_mean
?
IMobilenetV1/Conv2d_13_pointwise/BatchNorm/moving_mean/Read/ReadVariableOpReadVariableOp5MobilenetV1/Conv2d_13_pointwise/BatchNorm/moving_mean*
_output_shapes	
:?*
dtype0
?
8MobilenetV1/Conv2d_7_depthwise/BatchNorm/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*I
shared_name:8MobilenetV1/Conv2d_7_depthwise/BatchNorm/moving_variance
?
LMobilenetV1/Conv2d_7_depthwise/BatchNorm/moving_variance/Read/ReadVariableOpReadVariableOp8MobilenetV1/Conv2d_7_depthwise/BatchNorm/moving_variance*
_output_shapes	
:?*
dtype0
?
&MobilenetV1/Conv2d_7_pointwise/weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*7
shared_name(&MobilenetV1/Conv2d_7_pointwise/weights
?
:MobilenetV1/Conv2d_7_pointwise/weights/Read/ReadVariableOpReadVariableOp&MobilenetV1/Conv2d_7_pointwise/weights*(
_output_shapes
:??*
dtype0
?
&MobilenetV1/Conv2d_5_pointwise/weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*7
shared_name(&MobilenetV1/Conv2d_5_pointwise/weights
?
:MobilenetV1/Conv2d_5_pointwise/weights/Read/ReadVariableOpReadVariableOp&MobilenetV1/Conv2d_5_pointwise/weights*(
_output_shapes
:??*
dtype0
?
8MobilenetV1/Conv2d_1_pointwise/BatchNorm/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*I
shared_name:8MobilenetV1/Conv2d_1_pointwise/BatchNorm/moving_variance
?
LMobilenetV1/Conv2d_1_pointwise/BatchNorm/moving_variance/Read/ReadVariableOpReadVariableOp8MobilenetV1/Conv2d_1_pointwise/BatchNorm/moving_variance*
_output_shapes
:0*
dtype0
?
4MobilenetV1/Conv2d_1_depthwise/BatchNorm/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*E
shared_name64MobilenetV1/Conv2d_1_depthwise/BatchNorm/moving_mean
?
HMobilenetV1/Conv2d_1_depthwise/BatchNorm/moving_mean/Read/ReadVariableOpReadVariableOp4MobilenetV1/Conv2d_1_depthwise/BatchNorm/moving_mean*
_output_shapes
:*
dtype0
?
8MobilenetV1/Conv2d_3_pointwise/BatchNorm/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*I
shared_name:8MobilenetV1/Conv2d_3_pointwise/BatchNorm/moving_variance
?
LMobilenetV1/Conv2d_3_pointwise/BatchNorm/moving_variance/Read/ReadVariableOpReadVariableOp8MobilenetV1/Conv2d_3_pointwise/BatchNorm/moving_variance*
_output_shapes
:`*
dtype0
?
4MobilenetV1/Conv2d_9_pointwise/BatchNorm/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*E
shared_name64MobilenetV1/Conv2d_9_pointwise/BatchNorm/moving_mean
?
HMobilenetV1/Conv2d_9_pointwise/BatchNorm/moving_mean/Read/ReadVariableOpReadVariableOp4MobilenetV1/Conv2d_9_pointwise/BatchNorm/moving_mean*
_output_shapes	
:?*
dtype0
?
.MobilenetV1/Conv2d_10_pointwise/BatchNorm/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*?
shared_name0.MobilenetV1/Conv2d_10_pointwise/BatchNorm/beta
?
BMobilenetV1/Conv2d_10_pointwise/BatchNorm/beta/Read/ReadVariableOpReadVariableOp.MobilenetV1/Conv2d_10_pointwise/BatchNorm/beta*
_output_shapes	
:?*
dtype0
?
9MobilenetV1/Conv2d_12_depthwise/BatchNorm/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*J
shared_name;9MobilenetV1/Conv2d_12_depthwise/BatchNorm/moving_variance
?
MMobilenetV1/Conv2d_12_depthwise/BatchNorm/moving_variance/Read/ReadVariableOpReadVariableOp9MobilenetV1/Conv2d_12_depthwise/BatchNorm/moving_variance*
_output_shapes	
:?*
dtype0
?
4MobilenetV1/Conv2d_4_pointwise/BatchNorm/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*E
shared_name64MobilenetV1/Conv2d_4_pointwise/BatchNorm/moving_mean
?
HMobilenetV1/Conv2d_4_pointwise/BatchNorm/moving_mean/Read/ReadVariableOpReadVariableOp4MobilenetV1/Conv2d_4_pointwise/BatchNorm/moving_mean*
_output_shapes	
:?*
dtype0
?
'MobilenetV1/Conv2d_10_pointwise/weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*8
shared_name)'MobilenetV1/Conv2d_10_pointwise/weights
?
;MobilenetV1/Conv2d_10_pointwise/weights/Read/ReadVariableOpReadVariableOp'MobilenetV1/Conv2d_10_pointwise/weights*(
_output_shapes
:??*
dtype0
?
0MobilenetV1/Conv2d_1_depthwise/depthwise_weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*A
shared_name20MobilenetV1/Conv2d_1_depthwise/depthwise_weights
?
DMobilenetV1/Conv2d_1_depthwise/depthwise_weights/Read/ReadVariableOpReadVariableOp0MobilenetV1/Conv2d_1_depthwise/depthwise_weights*&
_output_shapes
:*
dtype0
?
-MobilenetV1/Conv2d_1_pointwise/BatchNorm/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*>
shared_name/-MobilenetV1/Conv2d_1_pointwise/BatchNorm/beta
?
AMobilenetV1/Conv2d_1_pointwise/BatchNorm/beta/Read/ReadVariableOpReadVariableOp-MobilenetV1/Conv2d_1_pointwise/BatchNorm/beta*
_output_shapes
:0*
dtype0
?
.MobilenetV1/Conv2d_0/BatchNorm/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*?
shared_name0.MobilenetV1/Conv2d_0/BatchNorm/moving_variance
?
BMobilenetV1/Conv2d_0/BatchNorm/moving_variance/Read/ReadVariableOpReadVariableOp.MobilenetV1/Conv2d_0/BatchNorm/moving_variance*
_output_shapes
:*
dtype0
?
.MobilenetV1/Conv2d_2_pointwise/BatchNorm/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*?
shared_name0.MobilenetV1/Conv2d_2_pointwise/BatchNorm/gamma
?
BMobilenetV1/Conv2d_2_pointwise/BatchNorm/gamma/Read/ReadVariableOpReadVariableOp.MobilenetV1/Conv2d_2_pointwise/BatchNorm/gamma*
_output_shapes
:`*
dtype0
?
0MobilenetV1/Conv2d_9_depthwise/depthwise_weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*A
shared_name20MobilenetV1/Conv2d_9_depthwise/depthwise_weights
?
DMobilenetV1/Conv2d_9_depthwise/depthwise_weights/Read/ReadVariableOpReadVariableOp0MobilenetV1/Conv2d_9_depthwise/depthwise_weights*'
_output_shapes
:?*
dtype0
?
5MobilenetV1/Conv2d_11_pointwise/BatchNorm/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*F
shared_name75MobilenetV1/Conv2d_11_pointwise/BatchNorm/moving_mean
?
IMobilenetV1/Conv2d_11_pointwise/BatchNorm/moving_mean/Read/ReadVariableOpReadVariableOp5MobilenetV1/Conv2d_11_pointwise/BatchNorm/moving_mean*
_output_shapes	
:?*
dtype0
?
0MobilenetV1/Conv2d_8_depthwise/depthwise_weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*A
shared_name20MobilenetV1/Conv2d_8_depthwise/depthwise_weights
?
DMobilenetV1/Conv2d_8_depthwise/depthwise_weights/Read/ReadVariableOpReadVariableOp0MobilenetV1/Conv2d_8_depthwise/depthwise_weights*'
_output_shapes
:?*
dtype0
?
-MobilenetV1/Conv2d_8_depthwise/BatchNorm/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*>
shared_name/-MobilenetV1/Conv2d_8_depthwise/BatchNorm/beta
?
AMobilenetV1/Conv2d_8_depthwise/BatchNorm/beta/Read/ReadVariableOpReadVariableOp-MobilenetV1/Conv2d_8_depthwise/BatchNorm/beta*
_output_shapes	
:?*
dtype0
?
8MobilenetV1/Conv2d_8_pointwise/BatchNorm/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*I
shared_name:8MobilenetV1/Conv2d_8_pointwise/BatchNorm/moving_variance
?
LMobilenetV1/Conv2d_8_pointwise/BatchNorm/moving_variance/Read/ReadVariableOpReadVariableOp8MobilenetV1/Conv2d_8_pointwise/BatchNorm/moving_variance*
_output_shapes	
:?*
dtype0
?
&MobilenetV1/Conv2d_3_pointwise/weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape:``*7
shared_name(&MobilenetV1/Conv2d_3_pointwise/weights
?
:MobilenetV1/Conv2d_3_pointwise/weights/Read/ReadVariableOpReadVariableOp&MobilenetV1/Conv2d_3_pointwise/weights*&
_output_shapes
:``*
dtype0
?
.MobilenetV1/Conv2d_12_depthwise/BatchNorm/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*?
shared_name0.MobilenetV1/Conv2d_12_depthwise/BatchNorm/beta
?
BMobilenetV1/Conv2d_12_depthwise/BatchNorm/beta/Read/ReadVariableOpReadVariableOp.MobilenetV1/Conv2d_12_depthwise/BatchNorm/beta*
_output_shapes	
:?*
dtype0
?
9MobilenetV1/Conv2d_13_depthwise/BatchNorm/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*J
shared_name;9MobilenetV1/Conv2d_13_depthwise/BatchNorm/moving_variance
?
MMobilenetV1/Conv2d_13_depthwise/BatchNorm/moving_variance/Read/ReadVariableOpReadVariableOp9MobilenetV1/Conv2d_13_depthwise/BatchNorm/moving_variance*
_output_shapes	
:?*
dtype0
?
'MobilenetV1/Conv2d_13_pointwise/weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*8
shared_name)'MobilenetV1/Conv2d_13_pointwise/weights
?
;MobilenetV1/Conv2d_13_pointwise/weights/Read/ReadVariableOpReadVariableOp'MobilenetV1/Conv2d_13_pointwise/weights*(
_output_shapes
:??*
dtype0
?
0MobilenetV1/Conv2d_7_depthwise/depthwise_weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*A
shared_name20MobilenetV1/Conv2d_7_depthwise/depthwise_weights
?
DMobilenetV1/Conv2d_7_depthwise/depthwise_weights/Read/ReadVariableOpReadVariableOp0MobilenetV1/Conv2d_7_depthwise/depthwise_weights*'
_output_shapes
:?*
dtype0
?
-MobilenetV1/Conv2d_7_depthwise/BatchNorm/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*>
shared_name/-MobilenetV1/Conv2d_7_depthwise/BatchNorm/beta
?
AMobilenetV1/Conv2d_7_depthwise/BatchNorm/beta/Read/ReadVariableOpReadVariableOp-MobilenetV1/Conv2d_7_depthwise/BatchNorm/beta*
_output_shapes	
:?*
dtype0
?
/MobilenetV1/Conv2d_13_pointwise/BatchNorm/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*@
shared_name1/MobilenetV1/Conv2d_13_pointwise/BatchNorm/gamma
?
CMobilenetV1/Conv2d_13_pointwise/BatchNorm/gamma/Read/ReadVariableOpReadVariableOp/MobilenetV1/Conv2d_13_pointwise/BatchNorm/gamma*
_output_shapes	
:?*
dtype0
?
$MobilenetV1/Conv2d_0/BatchNorm/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$MobilenetV1/Conv2d_0/BatchNorm/gamma
?
8MobilenetV1/Conv2d_0/BatchNorm/gamma/Read/ReadVariableOpReadVariableOp$MobilenetV1/Conv2d_0/BatchNorm/gamma*
_output_shapes
:*
dtype0
?
-MobilenetV1/Conv2d_4_depthwise/BatchNorm/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*>
shared_name/-MobilenetV1/Conv2d_4_depthwise/BatchNorm/beta
?
AMobilenetV1/Conv2d_4_depthwise/BatchNorm/beta/Read/ReadVariableOpReadVariableOp-MobilenetV1/Conv2d_4_depthwise/BatchNorm/beta*
_output_shapes
:`*
dtype0
?
-MobilenetV1/Conv2d_9_pointwise/BatchNorm/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*>
shared_name/-MobilenetV1/Conv2d_9_pointwise/BatchNorm/beta
?
AMobilenetV1/Conv2d_9_pointwise/BatchNorm/beta/Read/ReadVariableOpReadVariableOp-MobilenetV1/Conv2d_9_pointwise/BatchNorm/beta*
_output_shapes	
:?*
dtype0
?
.MobilenetV1/Conv2d_5_pointwise/BatchNorm/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*?
shared_name0.MobilenetV1/Conv2d_5_pointwise/BatchNorm/gamma
?
BMobilenetV1/Conv2d_5_pointwise/BatchNorm/gamma/Read/ReadVariableOpReadVariableOp.MobilenetV1/Conv2d_5_pointwise/BatchNorm/gamma*
_output_shapes	
:?*
dtype0
?
.MobilenetV1/Conv2d_2_depthwise/BatchNorm/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*?
shared_name0.MobilenetV1/Conv2d_2_depthwise/BatchNorm/gamma
?
BMobilenetV1/Conv2d_2_depthwise/BatchNorm/gamma/Read/ReadVariableOpReadVariableOp.MobilenetV1/Conv2d_2_depthwise/BatchNorm/gamma*
_output_shapes
:0*
dtype0
?
.MobilenetV1/Conv2d_4_pointwise/BatchNorm/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*?
shared_name0.MobilenetV1/Conv2d_4_pointwise/BatchNorm/gamma
?
BMobilenetV1/Conv2d_4_pointwise/BatchNorm/gamma/Read/ReadVariableOpReadVariableOp.MobilenetV1/Conv2d_4_pointwise/BatchNorm/gamma*
_output_shapes	
:?*
dtype0
?
0MobilenetV1/Conv2d_6_depthwise/depthwise_weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*A
shared_name20MobilenetV1/Conv2d_6_depthwise/depthwise_weights
?
DMobilenetV1/Conv2d_6_depthwise/depthwise_weights/Read/ReadVariableOpReadVariableOp0MobilenetV1/Conv2d_6_depthwise/depthwise_weights*'
_output_shapes
:?*
dtype0
?
.MobilenetV1/Conv2d_9_depthwise/BatchNorm/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*?
shared_name0.MobilenetV1/Conv2d_9_depthwise/BatchNorm/gamma
?
BMobilenetV1/Conv2d_9_depthwise/BatchNorm/gamma/Read/ReadVariableOpReadVariableOp.MobilenetV1/Conv2d_9_depthwise/BatchNorm/gamma*
_output_shapes	
:?*
dtype0
?
9MobilenetV1/Conv2d_11_depthwise/BatchNorm/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*J
shared_name;9MobilenetV1/Conv2d_11_depthwise/BatchNorm/moving_variance
?
MMobilenetV1/Conv2d_11_depthwise/BatchNorm/moving_variance/Read/ReadVariableOpReadVariableOp9MobilenetV1/Conv2d_11_depthwise/BatchNorm/moving_variance*
_output_shapes	
:?*
dtype0
?
/MobilenetV1/Conv2d_13_depthwise/BatchNorm/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*@
shared_name1/MobilenetV1/Conv2d_13_depthwise/BatchNorm/gamma
?
CMobilenetV1/Conv2d_13_depthwise/BatchNorm/gamma/Read/ReadVariableOpReadVariableOp/MobilenetV1/Conv2d_13_depthwise/BatchNorm/gamma*
_output_shapes	
:?*
dtype0
?
9MobilenetV1/Conv2d_13_pointwise/BatchNorm/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*J
shared_name;9MobilenetV1/Conv2d_13_pointwise/BatchNorm/moving_variance
?
MMobilenetV1/Conv2d_13_pointwise/BatchNorm/moving_variance/Read/ReadVariableOpReadVariableOp9MobilenetV1/Conv2d_13_pointwise/BatchNorm/moving_variance*
_output_shapes	
:?*
dtype0
?
5MobilenetV1/Conv2d_11_depthwise/BatchNorm/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*F
shared_name75MobilenetV1/Conv2d_11_depthwise/BatchNorm/moving_mean
?
IMobilenetV1/Conv2d_11_depthwise/BatchNorm/moving_mean/Read/ReadVariableOpReadVariableOp5MobilenetV1/Conv2d_11_depthwise/BatchNorm/moving_mean*
_output_shapes	
:?*
dtype0
?
&MobilenetV1/Conv2d_8_pointwise/weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*7
shared_name(&MobilenetV1/Conv2d_8_pointwise/weights
?
:MobilenetV1/Conv2d_8_pointwise/weights/Read/ReadVariableOpReadVariableOp&MobilenetV1/Conv2d_8_pointwise/weights*(
_output_shapes
:??*
dtype0
?
4MobilenetV1/Conv2d_3_depthwise/BatchNorm/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*E
shared_name64MobilenetV1/Conv2d_3_depthwise/BatchNorm/moving_mean
?
HMobilenetV1/Conv2d_3_depthwise/BatchNorm/moving_mean/Read/ReadVariableOpReadVariableOp4MobilenetV1/Conv2d_3_depthwise/BatchNorm/moving_mean*
_output_shapes
:`*
dtype0
?
&MobilenetV1/Conv2d_4_pointwise/weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape:`?*7
shared_name(&MobilenetV1/Conv2d_4_pointwise/weights
?
:MobilenetV1/Conv2d_4_pointwise/weights/Read/ReadVariableOpReadVariableOp&MobilenetV1/Conv2d_4_pointwise/weights*'
_output_shapes
:`?*
dtype0
?
8MobilenetV1/Conv2d_4_pointwise/BatchNorm/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*I
shared_name:8MobilenetV1/Conv2d_4_pointwise/BatchNorm/moving_variance
?
LMobilenetV1/Conv2d_4_pointwise/BatchNorm/moving_variance/Read/ReadVariableOpReadVariableOp8MobilenetV1/Conv2d_4_pointwise/BatchNorm/moving_variance*
_output_shapes	
:?*
dtype0
?
-MobilenetV1/Conv2d_6_depthwise/BatchNorm/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*>
shared_name/-MobilenetV1/Conv2d_6_depthwise/BatchNorm/beta
?
AMobilenetV1/Conv2d_6_depthwise/BatchNorm/beta/Read/ReadVariableOpReadVariableOp-MobilenetV1/Conv2d_6_depthwise/BatchNorm/beta*
_output_shapes	
:?*
dtype0
?
8MobilenetV1/Conv2d_6_pointwise/BatchNorm/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*I
shared_name:8MobilenetV1/Conv2d_6_pointwise/BatchNorm/moving_variance
?
LMobilenetV1/Conv2d_6_pointwise/BatchNorm/moving_variance/Read/ReadVariableOpReadVariableOp8MobilenetV1/Conv2d_6_pointwise/BatchNorm/moving_variance*
_output_shapes	
:?*
dtype0
?
4MobilenetV1/Conv2d_9_depthwise/BatchNorm/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*E
shared_name64MobilenetV1/Conv2d_9_depthwise/BatchNorm/moving_mean
?
HMobilenetV1/Conv2d_9_depthwise/BatchNorm/moving_mean/Read/ReadVariableOpReadVariableOp4MobilenetV1/Conv2d_9_depthwise/BatchNorm/moving_mean*
_output_shapes	
:?*
dtype0
?
.MobilenetV1/Conv2d_9_pointwise/BatchNorm/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*?
shared_name0.MobilenetV1/Conv2d_9_pointwise/BatchNorm/gamma
?
BMobilenetV1/Conv2d_9_pointwise/BatchNorm/gamma/Read/ReadVariableOpReadVariableOp.MobilenetV1/Conv2d_9_pointwise/BatchNorm/gamma*
_output_shapes	
:?*
dtype0
?
5MobilenetV1/Conv2d_12_pointwise/BatchNorm/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*F
shared_name75MobilenetV1/Conv2d_12_pointwise/BatchNorm/moving_mean
?
IMobilenetV1/Conv2d_12_pointwise/BatchNorm/moving_mean/Read/ReadVariableOpReadVariableOp5MobilenetV1/Conv2d_12_pointwise/BatchNorm/moving_mean*
_output_shapes	
:?*
dtype0
?
.MobilenetV1/Conv2d_13_pointwise/BatchNorm/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*?
shared_name0.MobilenetV1/Conv2d_13_pointwise/BatchNorm/beta
?
BMobilenetV1/Conv2d_13_pointwise/BatchNorm/beta/Read/ReadVariableOpReadVariableOp.MobilenetV1/Conv2d_13_pointwise/BatchNorm/beta*
_output_shapes	
:?*
dtype0
?
4MobilenetV1/Conv2d_7_pointwise/BatchNorm/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*E
shared_name64MobilenetV1/Conv2d_7_pointwise/BatchNorm/moving_mean
?
HMobilenetV1/Conv2d_7_pointwise/BatchNorm/moving_mean/Read/ReadVariableOpReadVariableOp4MobilenetV1/Conv2d_7_pointwise/BatchNorm/moving_mean*
_output_shapes	
:?*
dtype0
?
5MobilenetV1/Conv2d_12_depthwise/BatchNorm/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*F
shared_name75MobilenetV1/Conv2d_12_depthwise/BatchNorm/moving_mean
?
IMobilenetV1/Conv2d_12_depthwise/BatchNorm/moving_mean/Read/ReadVariableOpReadVariableOp5MobilenetV1/Conv2d_12_depthwise/BatchNorm/moving_mean*
_output_shapes	
:?*
dtype0
?
0MobilenetV1/Conv2d_3_depthwise/depthwise_weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*A
shared_name20MobilenetV1/Conv2d_3_depthwise/depthwise_weights
?
DMobilenetV1/Conv2d_3_depthwise/depthwise_weights/Read/ReadVariableOpReadVariableOp0MobilenetV1/Conv2d_3_depthwise/depthwise_weights*&
_output_shapes
:`*
dtype0
?
-MobilenetV1/Conv2d_4_pointwise/BatchNorm/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*>
shared_name/-MobilenetV1/Conv2d_4_pointwise/BatchNorm/beta
?
AMobilenetV1/Conv2d_4_pointwise/BatchNorm/beta/Read/ReadVariableOpReadVariableOp-MobilenetV1/Conv2d_4_pointwise/BatchNorm/beta*
_output_shapes	
:?*
dtype0
?
4MobilenetV1/Conv2d_2_depthwise/BatchNorm/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*E
shared_name64MobilenetV1/Conv2d_2_depthwise/BatchNorm/moving_mean
?
HMobilenetV1/Conv2d_2_depthwise/BatchNorm/moving_mean/Read/ReadVariableOpReadVariableOp4MobilenetV1/Conv2d_2_depthwise/BatchNorm/moving_mean*
_output_shapes
:0*
dtype0
?
-MobilenetV1/Conv2d_2_pointwise/BatchNorm/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*>
shared_name/-MobilenetV1/Conv2d_2_pointwise/BatchNorm/beta
?
AMobilenetV1/Conv2d_2_pointwise/BatchNorm/beta/Read/ReadVariableOpReadVariableOp-MobilenetV1/Conv2d_2_pointwise/BatchNorm/beta*
_output_shapes
:`*
dtype0
?
/MobilenetV1/Conv2d_10_depthwise/BatchNorm/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*@
shared_name1/MobilenetV1/Conv2d_10_depthwise/BatchNorm/gamma
?
CMobilenetV1/Conv2d_10_depthwise/BatchNorm/gamma/Read/ReadVariableOpReadVariableOp/MobilenetV1/Conv2d_10_depthwise/BatchNorm/gamma*
_output_shapes	
:?*
dtype0
?
1MobilenetV1/Conv2d_12_depthwise/depthwise_weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*B
shared_name31MobilenetV1/Conv2d_12_depthwise/depthwise_weights
?
EMobilenetV1/Conv2d_12_depthwise/depthwise_weights/Read/ReadVariableOpReadVariableOp1MobilenetV1/Conv2d_12_depthwise/depthwise_weights*'
_output_shapes
:?*
dtype0
?
#MobilenetV1/Conv2d_0/BatchNorm/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#MobilenetV1/Conv2d_0/BatchNorm/beta
?
7MobilenetV1/Conv2d_0/BatchNorm/beta/Read/ReadVariableOpReadVariableOp#MobilenetV1/Conv2d_0/BatchNorm/beta*
_output_shapes
:*
dtype0
?
1MobilenetV1/Conv2d_11_depthwise/depthwise_weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*B
shared_name31MobilenetV1/Conv2d_11_depthwise/depthwise_weights
?
EMobilenetV1/Conv2d_11_depthwise/depthwise_weights/Read/ReadVariableOpReadVariableOp1MobilenetV1/Conv2d_11_depthwise/depthwise_weights*'
_output_shapes
:?*
dtype0
?
.MobilenetV1/Conv2d_11_depthwise/BatchNorm/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*?
shared_name0.MobilenetV1/Conv2d_11_depthwise/BatchNorm/beta
?
BMobilenetV1/Conv2d_11_depthwise/BatchNorm/beta/Read/ReadVariableOpReadVariableOp.MobilenetV1/Conv2d_11_depthwise/BatchNorm/beta*
_output_shapes	
:?*
dtype0
?
9MobilenetV1/Conv2d_11_pointwise/BatchNorm/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*J
shared_name;9MobilenetV1/Conv2d_11_pointwise/BatchNorm/moving_variance
?
MMobilenetV1/Conv2d_11_pointwise/BatchNorm/moving_variance/Read/ReadVariableOpReadVariableOp9MobilenetV1/Conv2d_11_pointwise/BatchNorm/moving_variance*
_output_shapes	
:?*
dtype0
?
4MobilenetV1/Conv2d_7_depthwise/BatchNorm/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*E
shared_name64MobilenetV1/Conv2d_7_depthwise/BatchNorm/moving_mean
?
HMobilenetV1/Conv2d_7_depthwise/BatchNorm/moving_mean/Read/ReadVariableOpReadVariableOp4MobilenetV1/Conv2d_7_depthwise/BatchNorm/moving_mean*
_output_shapes	
:?*
dtype0
?
.MobilenetV1/Conv2d_1_depthwise/BatchNorm/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*?
shared_name0.MobilenetV1/Conv2d_1_depthwise/BatchNorm/gamma
?
BMobilenetV1/Conv2d_1_depthwise/BatchNorm/gamma/Read/ReadVariableOpReadVariableOp.MobilenetV1/Conv2d_1_depthwise/BatchNorm/gamma*
_output_shapes
:*
dtype0
?
8MobilenetV1/Conv2d_3_depthwise/BatchNorm/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*I
shared_name:8MobilenetV1/Conv2d_3_depthwise/BatchNorm/moving_variance
?
LMobilenetV1/Conv2d_3_depthwise/BatchNorm/moving_variance/Read/ReadVariableOpReadVariableOp8MobilenetV1/Conv2d_3_depthwise/BatchNorm/moving_variance*
_output_shapes
:`*
dtype0
?
4MobilenetV1/Conv2d_5_pointwise/BatchNorm/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*E
shared_name64MobilenetV1/Conv2d_5_pointwise/BatchNorm/moving_mean
?
HMobilenetV1/Conv2d_5_pointwise/BatchNorm/moving_mean/Read/ReadVariableOpReadVariableOp4MobilenetV1/Conv2d_5_pointwise/BatchNorm/moving_mean*
_output_shapes	
:?*
dtype0
?
/MobilenetV1/Conv2d_12_pointwise/BatchNorm/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*@
shared_name1/MobilenetV1/Conv2d_12_pointwise/BatchNorm/gamma
?
CMobilenetV1/Conv2d_12_pointwise/BatchNorm/gamma/Read/ReadVariableOpReadVariableOp/MobilenetV1/Conv2d_12_pointwise/BatchNorm/gamma*
_output_shapes	
:?*
dtype0
?
&MobilenetV1/Conv2d_2_pointwise/weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape:0`*7
shared_name(&MobilenetV1/Conv2d_2_pointwise/weights
?
:MobilenetV1/Conv2d_2_pointwise/weights/Read/ReadVariableOpReadVariableOp&MobilenetV1/Conv2d_2_pointwise/weights*&
_output_shapes
:0`*
dtype0
?
8MobilenetV1/Conv2d_2_pointwise/BatchNorm/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*I
shared_name:8MobilenetV1/Conv2d_2_pointwise/BatchNorm/moving_variance
?
LMobilenetV1/Conv2d_2_pointwise/BatchNorm/moving_variance/Read/ReadVariableOpReadVariableOp8MobilenetV1/Conv2d_2_pointwise/BatchNorm/moving_variance*
_output_shapes
:`*
dtype0
?
.MobilenetV1/Conv2d_5_depthwise/BatchNorm/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*?
shared_name0.MobilenetV1/Conv2d_5_depthwise/BatchNorm/gamma
?
BMobilenetV1/Conv2d_5_depthwise/BatchNorm/gamma/Read/ReadVariableOpReadVariableOp.MobilenetV1/Conv2d_5_depthwise/BatchNorm/gamma*
_output_shapes	
:?*
dtype0
?
1MobilenetV1/Conv2d_10_depthwise/depthwise_weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*B
shared_name31MobilenetV1/Conv2d_10_depthwise/depthwise_weights
?
EMobilenetV1/Conv2d_10_depthwise/depthwise_weights/Read/ReadVariableOpReadVariableOp1MobilenetV1/Conv2d_10_depthwise/depthwise_weights*'
_output_shapes
:?*
dtype0
?
/MobilenetV1/Conv2d_11_pointwise/BatchNorm/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*@
shared_name1/MobilenetV1/Conv2d_11_pointwise/BatchNorm/gamma
?
CMobilenetV1/Conv2d_11_pointwise/BatchNorm/gamma/Read/ReadVariableOpReadVariableOp/MobilenetV1/Conv2d_11_pointwise/BatchNorm/gamma*
_output_shapes	
:?*
dtype0
?
.MobilenetV1/Conv2d_6_pointwise/BatchNorm/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*?
shared_name0.MobilenetV1/Conv2d_6_pointwise/BatchNorm/gamma
?
BMobilenetV1/Conv2d_6_pointwise/BatchNorm/gamma/Read/ReadVariableOpReadVariableOp.MobilenetV1/Conv2d_6_pointwise/BatchNorm/gamma*
_output_shapes	
:?*
dtype0

NoOpNoOp
??
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*??
value??B?? Bކ
e
	variables
trainable_variables
regularization_losses
save_counter

signatures
?
0
1
2
	3

4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
 26
!27
"28
#29
$30
%31
&32
'33
(34
)35
*36
+37
,38
-39
.40
/41
042
143
244
345
446
547
648
749
850
951
:52
;53
<54
=55
>56
?57
@58
A59
B60
C61
D62
E63
F64
G65
H66
I67
J68
K69
L70
M71
N72
O73
P74
Q75
R76
S77
T78
U79
V80
W81
X82
Y83
Z84
[85
\86
]87
^88
_89
`90
a91
b92
c93
d94
e95
f96
g97
h98
i99
j100
k101
l102
m103
n104
o105
p106
q107
r108
s109
t110
u111
v112
w113
x114
y115
z116
{117
|118
}119
~120
121
?122
?123
?124
?125
?126
?127
?128
?129
?130
?131
?132
?133
?134
?135
?136
?
0
1
	2

3
4
5
6
7
8
9
10
11
 12
!13
"14
#15
$16
%17
'18
(19
)20
*21
+22
-23
.24
025
226
327
528
729
830
:31
;32
<33
=34
>35
B36
D37
H38
I39
N40
Q41
R42
S43
U44
V45
X46
Y47
[48
\49
^50
_51
`52
a53
b54
c55
d56
e57
f58
g59
h60
i61
k62
n63
p64
r65
u66
w67
z68
{69
}70
~71
72
?73
?74
?75
?76
?77
?78
?79
?80
?81
?82
 
IG
VARIABLE_VALUEsave_counter'save_counter/.ATTRIBUTES/VARIABLE_VALUE
 
tr
VARIABLE_VALUE8MobilenetV1/Conv2d_1_depthwise/BatchNorm/moving_variance&variables/0/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE.MobilenetV1/Conv2d_3_depthwise/BatchNorm/gamma&variables/1/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE.MobilenetV1/Conv2d_7_pointwise/BatchNorm/gamma&variables/2/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUE-MobilenetV1/Conv2d_8_pointwise/BatchNorm/beta&variables/3/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE.MobilenetV1/Conv2d_11_pointwise/BatchNorm/beta&variables/4/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUE0MobilenetV1/Conv2d_4_depthwise/depthwise_weights&variables/5/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUE(MobilenetV1/Logits/Conv2d_1c_1x1/weights&variables/6/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUE*MobilenetV1/Conv2d_0/BatchNorm/moving_mean&variables/7/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUE4MobilenetV1/Conv2d_3_pointwise/BatchNorm/moving_mean&variables/8/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUE4MobilenetV1/Conv2d_4_depthwise/BatchNorm/moving_mean&variables/9/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUE8MobilenetV1/Conv2d_5_depthwise/BatchNorm/moving_variance'variables/10/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE.MobilenetV1/Conv2d_6_depthwise/BatchNorm/gamma'variables/11/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUE8MobilenetV1/Conv2d_7_pointwise/BatchNorm/moving_variance'variables/12/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUE8MobilenetV1/Conv2d_9_depthwise/BatchNorm/moving_variance'variables/13/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUE8MobilenetV1/Conv2d_9_pointwise/BatchNorm/moving_variance'variables/14/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUE'MobilenetV1/Conv2d_11_pointwise/weights'variables/15/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUE4MobilenetV1/Conv2d_1_pointwise/BatchNorm/moving_mean'variables/16/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUE4MobilenetV1/Conv2d_5_depthwise/BatchNorm/moving_mean'variables/17/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE5MobilenetV1/Conv2d_10_pointwise/BatchNorm/moving_mean'variables/18/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE-MobilenetV1/Conv2d_7_pointwise/BatchNorm/beta'variables/19/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEMobilenetV1/Conv2d_0/weights'variables/20/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUE8MobilenetV1/Conv2d_2_depthwise/BatchNorm/moving_variance'variables/21/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUE8MobilenetV1/Conv2d_4_depthwise/BatchNorm/moving_variance'variables/22/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUE4MobilenetV1/Conv2d_6_pointwise/BatchNorm/moving_mean'variables/23/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE-MobilenetV1/Conv2d_9_depthwise/BatchNorm/beta'variables/24/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUE/MobilenetV1/Conv2d_11_depthwise/BatchNorm/gamma'variables/25/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE&MobilenetV1/Conv2d_6_pointwise/weights'variables/26/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE-MobilenetV1/Conv2d_6_pointwise/BatchNorm/beta'variables/27/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE-MobilenetV1/Conv2d_1_depthwise/BatchNorm/beta'variables/28/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE.MobilenetV1/Conv2d_3_pointwise/BatchNorm/gamma'variables/29/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE.MobilenetV1/Conv2d_10_depthwise/BatchNorm/beta'variables/30/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUE'MobilenetV1/Logits/Conv2d_1c_1x1/biases'variables/31/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUE4MobilenetV1/Conv2d_8_pointwise/BatchNorm/moving_mean'variables/32/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE-MobilenetV1/Conv2d_5_pointwise/BatchNorm/beta'variables/33/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUE0MobilenetV1/Conv2d_2_depthwise/depthwise_weights'variables/34/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE&MobilenetV1/Conv2d_1_pointwise/weights'variables/35/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE.MobilenetV1/Conv2d_1_pointwise/BatchNorm/gamma'variables/36/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE-MobilenetV1/Conv2d_5_depthwise/BatchNorm/beta'variables/37/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUE8MobilenetV1/Conv2d_5_pointwise/BatchNorm/moving_variance'variables/38/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE.MobilenetV1/Conv2d_8_depthwise/BatchNorm/gamma'variables/39/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE1MobilenetV1/Conv2d_13_depthwise/depthwise_weights'variables/40/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUE8MobilenetV1/Conv2d_6_depthwise/BatchNorm/moving_variance'variables/41/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE&MobilenetV1/Conv2d_9_pointwise/weights'variables/42/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE9MobilenetV1/Conv2d_10_depthwise/BatchNorm/moving_variance'variables/43/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE.MobilenetV1/Conv2d_12_pointwise/BatchNorm/beta'variables/44/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE-MobilenetV1/Conv2d_2_depthwise/BatchNorm/beta'variables/45/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUE4MobilenetV1/Conv2d_2_pointwise/BatchNorm/moving_mean'variables/46/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE-MobilenetV1/Conv2d_3_pointwise/BatchNorm/beta'variables/47/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUE4MobilenetV1/Conv2d_6_depthwise/BatchNorm/moving_mean'variables/48/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUE/MobilenetV1/Conv2d_10_pointwise/BatchNorm/gamma'variables/49/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUE/MobilenetV1/Conv2d_12_depthwise/BatchNorm/gamma'variables/50/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE5MobilenetV1/Conv2d_13_depthwise/BatchNorm/moving_mean'variables/51/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUE0MobilenetV1/Conv2d_5_depthwise/depthwise_weights'variables/52/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE.MobilenetV1/Conv2d_13_depthwise/BatchNorm/beta'variables/53/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE.MobilenetV1/Conv2d_8_pointwise/BatchNorm/gamma'variables/54/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE-MobilenetV1/Conv2d_3_depthwise/BatchNorm/beta'variables/55/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE.MobilenetV1/Conv2d_7_depthwise/BatchNorm/gamma'variables/56/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUE8MobilenetV1/Conv2d_8_depthwise/BatchNorm/moving_variance'variables/57/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE5MobilenetV1/Conv2d_10_depthwise/BatchNorm/moving_mean'variables/58/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE9MobilenetV1/Conv2d_10_pointwise/BatchNorm/moving_variance'variables/59/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUE'MobilenetV1/Conv2d_12_pointwise/weights'variables/60/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE9MobilenetV1/Conv2d_12_pointwise/BatchNorm/moving_variance'variables/61/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE.MobilenetV1/Conv2d_4_depthwise/BatchNorm/gamma'variables/62/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUE4MobilenetV1/Conv2d_8_depthwise/BatchNorm/moving_mean'variables/63/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE5MobilenetV1/Conv2d_13_pointwise/BatchNorm/moving_mean'variables/64/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUE8MobilenetV1/Conv2d_7_depthwise/BatchNorm/moving_variance'variables/65/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE&MobilenetV1/Conv2d_7_pointwise/weights'variables/66/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE&MobilenetV1/Conv2d_5_pointwise/weights'variables/67/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUE8MobilenetV1/Conv2d_1_pointwise/BatchNorm/moving_variance'variables/68/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUE4MobilenetV1/Conv2d_1_depthwise/BatchNorm/moving_mean'variables/69/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUE8MobilenetV1/Conv2d_3_pointwise/BatchNorm/moving_variance'variables/70/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUE4MobilenetV1/Conv2d_9_pointwise/BatchNorm/moving_mean'variables/71/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE.MobilenetV1/Conv2d_10_pointwise/BatchNorm/beta'variables/72/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE9MobilenetV1/Conv2d_12_depthwise/BatchNorm/moving_variance'variables/73/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUE4MobilenetV1/Conv2d_4_pointwise/BatchNorm/moving_mean'variables/74/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUE'MobilenetV1/Conv2d_10_pointwise/weights'variables/75/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUE0MobilenetV1/Conv2d_1_depthwise/depthwise_weights'variables/76/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE-MobilenetV1/Conv2d_1_pointwise/BatchNorm/beta'variables/77/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE.MobilenetV1/Conv2d_0/BatchNorm/moving_variance'variables/78/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE.MobilenetV1/Conv2d_2_pointwise/BatchNorm/gamma'variables/79/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUE0MobilenetV1/Conv2d_9_depthwise/depthwise_weights'variables/80/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE5MobilenetV1/Conv2d_11_pointwise/BatchNorm/moving_mean'variables/81/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUE0MobilenetV1/Conv2d_8_depthwise/depthwise_weights'variables/82/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE-MobilenetV1/Conv2d_8_depthwise/BatchNorm/beta'variables/83/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUE8MobilenetV1/Conv2d_8_pointwise/BatchNorm/moving_variance'variables/84/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE&MobilenetV1/Conv2d_3_pointwise/weights'variables/85/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE.MobilenetV1/Conv2d_12_depthwise/BatchNorm/beta'variables/86/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE9MobilenetV1/Conv2d_13_depthwise/BatchNorm/moving_variance'variables/87/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUE'MobilenetV1/Conv2d_13_pointwise/weights'variables/88/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUE0MobilenetV1/Conv2d_7_depthwise/depthwise_weights'variables/89/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE-MobilenetV1/Conv2d_7_depthwise/BatchNorm/beta'variables/90/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUE/MobilenetV1/Conv2d_13_pointwise/BatchNorm/gamma'variables/91/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE$MobilenetV1/Conv2d_0/BatchNorm/gamma'variables/92/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE-MobilenetV1/Conv2d_4_depthwise/BatchNorm/beta'variables/93/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE-MobilenetV1/Conv2d_9_pointwise/BatchNorm/beta'variables/94/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE.MobilenetV1/Conv2d_5_pointwise/BatchNorm/gamma'variables/95/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE.MobilenetV1/Conv2d_2_depthwise/BatchNorm/gamma'variables/96/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE.MobilenetV1/Conv2d_4_pointwise/BatchNorm/gamma'variables/97/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUE0MobilenetV1/Conv2d_6_depthwise/depthwise_weights'variables/98/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE.MobilenetV1/Conv2d_9_depthwise/BatchNorm/gamma'variables/99/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUE9MobilenetV1/Conv2d_11_depthwise/BatchNorm/moving_variance(variables/100/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUE/MobilenetV1/Conv2d_13_depthwise/BatchNorm/gamma(variables/101/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUE9MobilenetV1/Conv2d_13_pointwise/BatchNorm/moving_variance(variables/102/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE5MobilenetV1/Conv2d_11_depthwise/BatchNorm/moving_mean(variables/103/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUE&MobilenetV1/Conv2d_8_pointwise/weights(variables/104/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE4MobilenetV1/Conv2d_3_depthwise/BatchNorm/moving_mean(variables/105/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUE&MobilenetV1/Conv2d_4_pointwise/weights(variables/106/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE8MobilenetV1/Conv2d_4_pointwise/BatchNorm/moving_variance(variables/107/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE-MobilenetV1/Conv2d_6_depthwise/BatchNorm/beta(variables/108/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE8MobilenetV1/Conv2d_6_pointwise/BatchNorm/moving_variance(variables/109/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE4MobilenetV1/Conv2d_9_depthwise/BatchNorm/moving_mean(variables/110/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUE.MobilenetV1/Conv2d_9_pointwise/BatchNorm/gamma(variables/111/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE5MobilenetV1/Conv2d_12_pointwise/BatchNorm/moving_mean(variables/112/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUE.MobilenetV1/Conv2d_13_pointwise/BatchNorm/beta(variables/113/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE4MobilenetV1/Conv2d_7_pointwise/BatchNorm/moving_mean(variables/114/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE5MobilenetV1/Conv2d_12_depthwise/BatchNorm/moving_mean(variables/115/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE0MobilenetV1/Conv2d_3_depthwise/depthwise_weights(variables/116/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE-MobilenetV1/Conv2d_4_pointwise/BatchNorm/beta(variables/117/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE4MobilenetV1/Conv2d_2_depthwise/BatchNorm/moving_mean(variables/118/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE-MobilenetV1/Conv2d_2_pointwise/BatchNorm/beta(variables/119/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUE/MobilenetV1/Conv2d_10_depthwise/BatchNorm/gamma(variables/120/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUE1MobilenetV1/Conv2d_12_depthwise/depthwise_weights(variables/121/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE#MobilenetV1/Conv2d_0/BatchNorm/beta(variables/122/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUE1MobilenetV1/Conv2d_11_depthwise/depthwise_weights(variables/123/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUE.MobilenetV1/Conv2d_11_depthwise/BatchNorm/beta(variables/124/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUE9MobilenetV1/Conv2d_11_pointwise/BatchNorm/moving_variance(variables/125/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE4MobilenetV1/Conv2d_7_depthwise/BatchNorm/moving_mean(variables/126/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUE.MobilenetV1/Conv2d_1_depthwise/BatchNorm/gamma(variables/127/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE8MobilenetV1/Conv2d_3_depthwise/BatchNorm/moving_variance(variables/128/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE4MobilenetV1/Conv2d_5_pointwise/BatchNorm/moving_mean(variables/129/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUE/MobilenetV1/Conv2d_12_pointwise/BatchNorm/gamma(variables/130/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUE&MobilenetV1/Conv2d_2_pointwise/weights(variables/131/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE8MobilenetV1/Conv2d_2_pointwise/BatchNorm/moving_variance(variables/132/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUE.MobilenetV1/Conv2d_5_depthwise/BatchNorm/gamma(variables/133/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUE1MobilenetV1/Conv2d_10_depthwise/depthwise_weights(variables/134/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUE/MobilenetV1/Conv2d_11_pointwise/BatchNorm/gamma(variables/135/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUE.MobilenetV1/Conv2d_6_pointwise/BatchNorm/gamma(variables/136/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_inputsPlaceholder*A
_output_shapes/
-:+???????????????????????????*
dtype0*6
shape-:+???????????????????????????
?:
StatefulPartitionedCallStatefulPartitionedCallserving_default_inputsMobilenetV1/Conv2d_0/weights$MobilenetV1/Conv2d_0/BatchNorm/gamma#MobilenetV1/Conv2d_0/BatchNorm/beta*MobilenetV1/Conv2d_0/BatchNorm/moving_mean.MobilenetV1/Conv2d_0/BatchNorm/moving_variance0MobilenetV1/Conv2d_1_depthwise/depthwise_weights.MobilenetV1/Conv2d_1_depthwise/BatchNorm/gamma-MobilenetV1/Conv2d_1_depthwise/BatchNorm/beta4MobilenetV1/Conv2d_1_depthwise/BatchNorm/moving_mean8MobilenetV1/Conv2d_1_depthwise/BatchNorm/moving_variance&MobilenetV1/Conv2d_1_pointwise/weights.MobilenetV1/Conv2d_1_pointwise/BatchNorm/gamma-MobilenetV1/Conv2d_1_pointwise/BatchNorm/beta4MobilenetV1/Conv2d_1_pointwise/BatchNorm/moving_mean8MobilenetV1/Conv2d_1_pointwise/BatchNorm/moving_variance0MobilenetV1/Conv2d_2_depthwise/depthwise_weights.MobilenetV1/Conv2d_2_depthwise/BatchNorm/gamma-MobilenetV1/Conv2d_2_depthwise/BatchNorm/beta4MobilenetV1/Conv2d_2_depthwise/BatchNorm/moving_mean8MobilenetV1/Conv2d_2_depthwise/BatchNorm/moving_variance&MobilenetV1/Conv2d_2_pointwise/weights.MobilenetV1/Conv2d_2_pointwise/BatchNorm/gamma-MobilenetV1/Conv2d_2_pointwise/BatchNorm/beta4MobilenetV1/Conv2d_2_pointwise/BatchNorm/moving_mean8MobilenetV1/Conv2d_2_pointwise/BatchNorm/moving_variance0MobilenetV1/Conv2d_3_depthwise/depthwise_weights.MobilenetV1/Conv2d_3_depthwise/BatchNorm/gamma-MobilenetV1/Conv2d_3_depthwise/BatchNorm/beta4MobilenetV1/Conv2d_3_depthwise/BatchNorm/moving_mean8MobilenetV1/Conv2d_3_depthwise/BatchNorm/moving_variance&MobilenetV1/Conv2d_3_pointwise/weights.MobilenetV1/Conv2d_3_pointwise/BatchNorm/gamma-MobilenetV1/Conv2d_3_pointwise/BatchNorm/beta4MobilenetV1/Conv2d_3_pointwise/BatchNorm/moving_mean8MobilenetV1/Conv2d_3_pointwise/BatchNorm/moving_variance0MobilenetV1/Conv2d_4_depthwise/depthwise_weights.MobilenetV1/Conv2d_4_depthwise/BatchNorm/gamma-MobilenetV1/Conv2d_4_depthwise/BatchNorm/beta4MobilenetV1/Conv2d_4_depthwise/BatchNorm/moving_mean8MobilenetV1/Conv2d_4_depthwise/BatchNorm/moving_variance&MobilenetV1/Conv2d_4_pointwise/weights.MobilenetV1/Conv2d_4_pointwise/BatchNorm/gamma-MobilenetV1/Conv2d_4_pointwise/BatchNorm/beta4MobilenetV1/Conv2d_4_pointwise/BatchNorm/moving_mean8MobilenetV1/Conv2d_4_pointwise/BatchNorm/moving_variance0MobilenetV1/Conv2d_5_depthwise/depthwise_weights.MobilenetV1/Conv2d_5_depthwise/BatchNorm/gamma-MobilenetV1/Conv2d_5_depthwise/BatchNorm/beta4MobilenetV1/Conv2d_5_depthwise/BatchNorm/moving_mean8MobilenetV1/Conv2d_5_depthwise/BatchNorm/moving_variance&MobilenetV1/Conv2d_5_pointwise/weights.MobilenetV1/Conv2d_5_pointwise/BatchNorm/gamma-MobilenetV1/Conv2d_5_pointwise/BatchNorm/beta4MobilenetV1/Conv2d_5_pointwise/BatchNorm/moving_mean8MobilenetV1/Conv2d_5_pointwise/BatchNorm/moving_variance0MobilenetV1/Conv2d_6_depthwise/depthwise_weights.MobilenetV1/Conv2d_6_depthwise/BatchNorm/gamma-MobilenetV1/Conv2d_6_depthwise/BatchNorm/beta4MobilenetV1/Conv2d_6_depthwise/BatchNorm/moving_mean8MobilenetV1/Conv2d_6_depthwise/BatchNorm/moving_variance&MobilenetV1/Conv2d_6_pointwise/weights.MobilenetV1/Conv2d_6_pointwise/BatchNorm/gamma-MobilenetV1/Conv2d_6_pointwise/BatchNorm/beta4MobilenetV1/Conv2d_6_pointwise/BatchNorm/moving_mean8MobilenetV1/Conv2d_6_pointwise/BatchNorm/moving_variance0MobilenetV1/Conv2d_7_depthwise/depthwise_weights.MobilenetV1/Conv2d_7_depthwise/BatchNorm/gamma-MobilenetV1/Conv2d_7_depthwise/BatchNorm/beta4MobilenetV1/Conv2d_7_depthwise/BatchNorm/moving_mean8MobilenetV1/Conv2d_7_depthwise/BatchNorm/moving_variance&MobilenetV1/Conv2d_7_pointwise/weights.MobilenetV1/Conv2d_7_pointwise/BatchNorm/gamma-MobilenetV1/Conv2d_7_pointwise/BatchNorm/beta4MobilenetV1/Conv2d_7_pointwise/BatchNorm/moving_mean8MobilenetV1/Conv2d_7_pointwise/BatchNorm/moving_variance0MobilenetV1/Conv2d_8_depthwise/depthwise_weights.MobilenetV1/Conv2d_8_depthwise/BatchNorm/gamma-MobilenetV1/Conv2d_8_depthwise/BatchNorm/beta4MobilenetV1/Conv2d_8_depthwise/BatchNorm/moving_mean8MobilenetV1/Conv2d_8_depthwise/BatchNorm/moving_variance&MobilenetV1/Conv2d_8_pointwise/weights.MobilenetV1/Conv2d_8_pointwise/BatchNorm/gamma-MobilenetV1/Conv2d_8_pointwise/BatchNorm/beta4MobilenetV1/Conv2d_8_pointwise/BatchNorm/moving_mean8MobilenetV1/Conv2d_8_pointwise/BatchNorm/moving_variance0MobilenetV1/Conv2d_9_depthwise/depthwise_weights.MobilenetV1/Conv2d_9_depthwise/BatchNorm/gamma-MobilenetV1/Conv2d_9_depthwise/BatchNorm/beta4MobilenetV1/Conv2d_9_depthwise/BatchNorm/moving_mean8MobilenetV1/Conv2d_9_depthwise/BatchNorm/moving_variance&MobilenetV1/Conv2d_9_pointwise/weights.MobilenetV1/Conv2d_9_pointwise/BatchNorm/gamma-MobilenetV1/Conv2d_9_pointwise/BatchNorm/beta4MobilenetV1/Conv2d_9_pointwise/BatchNorm/moving_mean8MobilenetV1/Conv2d_9_pointwise/BatchNorm/moving_variance1MobilenetV1/Conv2d_10_depthwise/depthwise_weights/MobilenetV1/Conv2d_10_depthwise/BatchNorm/gamma.MobilenetV1/Conv2d_10_depthwise/BatchNorm/beta5MobilenetV1/Conv2d_10_depthwise/BatchNorm/moving_mean9MobilenetV1/Conv2d_10_depthwise/BatchNorm/moving_variance'MobilenetV1/Conv2d_10_pointwise/weights/MobilenetV1/Conv2d_10_pointwise/BatchNorm/gamma.MobilenetV1/Conv2d_10_pointwise/BatchNorm/beta5MobilenetV1/Conv2d_10_pointwise/BatchNorm/moving_mean9MobilenetV1/Conv2d_10_pointwise/BatchNorm/moving_variance1MobilenetV1/Conv2d_11_depthwise/depthwise_weights/MobilenetV1/Conv2d_11_depthwise/BatchNorm/gamma.MobilenetV1/Conv2d_11_depthwise/BatchNorm/beta5MobilenetV1/Conv2d_11_depthwise/BatchNorm/moving_mean9MobilenetV1/Conv2d_11_depthwise/BatchNorm/moving_variance'MobilenetV1/Conv2d_11_pointwise/weights/MobilenetV1/Conv2d_11_pointwise/BatchNorm/gamma.MobilenetV1/Conv2d_11_pointwise/BatchNorm/beta5MobilenetV1/Conv2d_11_pointwise/BatchNorm/moving_mean9MobilenetV1/Conv2d_11_pointwise/BatchNorm/moving_variance1MobilenetV1/Conv2d_12_depthwise/depthwise_weights/MobilenetV1/Conv2d_12_depthwise/BatchNorm/gamma.MobilenetV1/Conv2d_12_depthwise/BatchNorm/beta5MobilenetV1/Conv2d_12_depthwise/BatchNorm/moving_mean9MobilenetV1/Conv2d_12_depthwise/BatchNorm/moving_variance'MobilenetV1/Conv2d_12_pointwise/weights/MobilenetV1/Conv2d_12_pointwise/BatchNorm/gamma.MobilenetV1/Conv2d_12_pointwise/BatchNorm/beta5MobilenetV1/Conv2d_12_pointwise/BatchNorm/moving_mean9MobilenetV1/Conv2d_12_pointwise/BatchNorm/moving_variance1MobilenetV1/Conv2d_13_depthwise/depthwise_weights/MobilenetV1/Conv2d_13_depthwise/BatchNorm/gamma.MobilenetV1/Conv2d_13_depthwise/BatchNorm/beta5MobilenetV1/Conv2d_13_depthwise/BatchNorm/moving_mean9MobilenetV1/Conv2d_13_depthwise/BatchNorm/moving_variance'MobilenetV1/Conv2d_13_pointwise/weights/MobilenetV1/Conv2d_13_pointwise/BatchNorm/gamma.MobilenetV1/Conv2d_13_pointwise/BatchNorm/beta5MobilenetV1/Conv2d_13_pointwise/BatchNorm/moving_mean9MobilenetV1/Conv2d_13_pointwise/BatchNorm/moving_variance(MobilenetV1/Logits/Conv2d_1c_1x1/weights'MobilenetV1/Logits/Conv2d_1c_1x1/biases*?
Tin?
?2?*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*?
_read_only_resource_inputs?
??	
 !"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~??????????*-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference_signature_wrapper_5765
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?N
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename save_counter/Read/ReadVariableOpLMobilenetV1/Conv2d_1_depthwise/BatchNorm/moving_variance/Read/ReadVariableOpBMobilenetV1/Conv2d_3_depthwise/BatchNorm/gamma/Read/ReadVariableOpBMobilenetV1/Conv2d_7_pointwise/BatchNorm/gamma/Read/ReadVariableOpAMobilenetV1/Conv2d_8_pointwise/BatchNorm/beta/Read/ReadVariableOpBMobilenetV1/Conv2d_11_pointwise/BatchNorm/beta/Read/ReadVariableOpDMobilenetV1/Conv2d_4_depthwise/depthwise_weights/Read/ReadVariableOp<MobilenetV1/Logits/Conv2d_1c_1x1/weights/Read/ReadVariableOp>MobilenetV1/Conv2d_0/BatchNorm/moving_mean/Read/ReadVariableOpHMobilenetV1/Conv2d_3_pointwise/BatchNorm/moving_mean/Read/ReadVariableOpHMobilenetV1/Conv2d_4_depthwise/BatchNorm/moving_mean/Read/ReadVariableOpLMobilenetV1/Conv2d_5_depthwise/BatchNorm/moving_variance/Read/ReadVariableOpBMobilenetV1/Conv2d_6_depthwise/BatchNorm/gamma/Read/ReadVariableOpLMobilenetV1/Conv2d_7_pointwise/BatchNorm/moving_variance/Read/ReadVariableOpLMobilenetV1/Conv2d_9_depthwise/BatchNorm/moving_variance/Read/ReadVariableOpLMobilenetV1/Conv2d_9_pointwise/BatchNorm/moving_variance/Read/ReadVariableOp;MobilenetV1/Conv2d_11_pointwise/weights/Read/ReadVariableOpHMobilenetV1/Conv2d_1_pointwise/BatchNorm/moving_mean/Read/ReadVariableOpHMobilenetV1/Conv2d_5_depthwise/BatchNorm/moving_mean/Read/ReadVariableOpIMobilenetV1/Conv2d_10_pointwise/BatchNorm/moving_mean/Read/ReadVariableOpAMobilenetV1/Conv2d_7_pointwise/BatchNorm/beta/Read/ReadVariableOp0MobilenetV1/Conv2d_0/weights/Read/ReadVariableOpLMobilenetV1/Conv2d_2_depthwise/BatchNorm/moving_variance/Read/ReadVariableOpLMobilenetV1/Conv2d_4_depthwise/BatchNorm/moving_variance/Read/ReadVariableOpHMobilenetV1/Conv2d_6_pointwise/BatchNorm/moving_mean/Read/ReadVariableOpAMobilenetV1/Conv2d_9_depthwise/BatchNorm/beta/Read/ReadVariableOpCMobilenetV1/Conv2d_11_depthwise/BatchNorm/gamma/Read/ReadVariableOp:MobilenetV1/Conv2d_6_pointwise/weights/Read/ReadVariableOpAMobilenetV1/Conv2d_6_pointwise/BatchNorm/beta/Read/ReadVariableOpAMobilenetV1/Conv2d_1_depthwise/BatchNorm/beta/Read/ReadVariableOpBMobilenetV1/Conv2d_3_pointwise/BatchNorm/gamma/Read/ReadVariableOpBMobilenetV1/Conv2d_10_depthwise/BatchNorm/beta/Read/ReadVariableOp;MobilenetV1/Logits/Conv2d_1c_1x1/biases/Read/ReadVariableOpHMobilenetV1/Conv2d_8_pointwise/BatchNorm/moving_mean/Read/ReadVariableOpAMobilenetV1/Conv2d_5_pointwise/BatchNorm/beta/Read/ReadVariableOpDMobilenetV1/Conv2d_2_depthwise/depthwise_weights/Read/ReadVariableOp:MobilenetV1/Conv2d_1_pointwise/weights/Read/ReadVariableOpBMobilenetV1/Conv2d_1_pointwise/BatchNorm/gamma/Read/ReadVariableOpAMobilenetV1/Conv2d_5_depthwise/BatchNorm/beta/Read/ReadVariableOpLMobilenetV1/Conv2d_5_pointwise/BatchNorm/moving_variance/Read/ReadVariableOpBMobilenetV1/Conv2d_8_depthwise/BatchNorm/gamma/Read/ReadVariableOpEMobilenetV1/Conv2d_13_depthwise/depthwise_weights/Read/ReadVariableOpLMobilenetV1/Conv2d_6_depthwise/BatchNorm/moving_variance/Read/ReadVariableOp:MobilenetV1/Conv2d_9_pointwise/weights/Read/ReadVariableOpMMobilenetV1/Conv2d_10_depthwise/BatchNorm/moving_variance/Read/ReadVariableOpBMobilenetV1/Conv2d_12_pointwise/BatchNorm/beta/Read/ReadVariableOpAMobilenetV1/Conv2d_2_depthwise/BatchNorm/beta/Read/ReadVariableOpHMobilenetV1/Conv2d_2_pointwise/BatchNorm/moving_mean/Read/ReadVariableOpAMobilenetV1/Conv2d_3_pointwise/BatchNorm/beta/Read/ReadVariableOpHMobilenetV1/Conv2d_6_depthwise/BatchNorm/moving_mean/Read/ReadVariableOpCMobilenetV1/Conv2d_10_pointwise/BatchNorm/gamma/Read/ReadVariableOpCMobilenetV1/Conv2d_12_depthwise/BatchNorm/gamma/Read/ReadVariableOpIMobilenetV1/Conv2d_13_depthwise/BatchNorm/moving_mean/Read/ReadVariableOpDMobilenetV1/Conv2d_5_depthwise/depthwise_weights/Read/ReadVariableOpBMobilenetV1/Conv2d_13_depthwise/BatchNorm/beta/Read/ReadVariableOpBMobilenetV1/Conv2d_8_pointwise/BatchNorm/gamma/Read/ReadVariableOpAMobilenetV1/Conv2d_3_depthwise/BatchNorm/beta/Read/ReadVariableOpBMobilenetV1/Conv2d_7_depthwise/BatchNorm/gamma/Read/ReadVariableOpLMobilenetV1/Conv2d_8_depthwise/BatchNorm/moving_variance/Read/ReadVariableOpIMobilenetV1/Conv2d_10_depthwise/BatchNorm/moving_mean/Read/ReadVariableOpMMobilenetV1/Conv2d_10_pointwise/BatchNorm/moving_variance/Read/ReadVariableOp;MobilenetV1/Conv2d_12_pointwise/weights/Read/ReadVariableOpMMobilenetV1/Conv2d_12_pointwise/BatchNorm/moving_variance/Read/ReadVariableOpBMobilenetV1/Conv2d_4_depthwise/BatchNorm/gamma/Read/ReadVariableOpHMobilenetV1/Conv2d_8_depthwise/BatchNorm/moving_mean/Read/ReadVariableOpIMobilenetV1/Conv2d_13_pointwise/BatchNorm/moving_mean/Read/ReadVariableOpLMobilenetV1/Conv2d_7_depthwise/BatchNorm/moving_variance/Read/ReadVariableOp:MobilenetV1/Conv2d_7_pointwise/weights/Read/ReadVariableOp:MobilenetV1/Conv2d_5_pointwise/weights/Read/ReadVariableOpLMobilenetV1/Conv2d_1_pointwise/BatchNorm/moving_variance/Read/ReadVariableOpHMobilenetV1/Conv2d_1_depthwise/BatchNorm/moving_mean/Read/ReadVariableOpLMobilenetV1/Conv2d_3_pointwise/BatchNorm/moving_variance/Read/ReadVariableOpHMobilenetV1/Conv2d_9_pointwise/BatchNorm/moving_mean/Read/ReadVariableOpBMobilenetV1/Conv2d_10_pointwise/BatchNorm/beta/Read/ReadVariableOpMMobilenetV1/Conv2d_12_depthwise/BatchNorm/moving_variance/Read/ReadVariableOpHMobilenetV1/Conv2d_4_pointwise/BatchNorm/moving_mean/Read/ReadVariableOp;MobilenetV1/Conv2d_10_pointwise/weights/Read/ReadVariableOpDMobilenetV1/Conv2d_1_depthwise/depthwise_weights/Read/ReadVariableOpAMobilenetV1/Conv2d_1_pointwise/BatchNorm/beta/Read/ReadVariableOpBMobilenetV1/Conv2d_0/BatchNorm/moving_variance/Read/ReadVariableOpBMobilenetV1/Conv2d_2_pointwise/BatchNorm/gamma/Read/ReadVariableOpDMobilenetV1/Conv2d_9_depthwise/depthwise_weights/Read/ReadVariableOpIMobilenetV1/Conv2d_11_pointwise/BatchNorm/moving_mean/Read/ReadVariableOpDMobilenetV1/Conv2d_8_depthwise/depthwise_weights/Read/ReadVariableOpAMobilenetV1/Conv2d_8_depthwise/BatchNorm/beta/Read/ReadVariableOpLMobilenetV1/Conv2d_8_pointwise/BatchNorm/moving_variance/Read/ReadVariableOp:MobilenetV1/Conv2d_3_pointwise/weights/Read/ReadVariableOpBMobilenetV1/Conv2d_12_depthwise/BatchNorm/beta/Read/ReadVariableOpMMobilenetV1/Conv2d_13_depthwise/BatchNorm/moving_variance/Read/ReadVariableOp;MobilenetV1/Conv2d_13_pointwise/weights/Read/ReadVariableOpDMobilenetV1/Conv2d_7_depthwise/depthwise_weights/Read/ReadVariableOpAMobilenetV1/Conv2d_7_depthwise/BatchNorm/beta/Read/ReadVariableOpCMobilenetV1/Conv2d_13_pointwise/BatchNorm/gamma/Read/ReadVariableOp8MobilenetV1/Conv2d_0/BatchNorm/gamma/Read/ReadVariableOpAMobilenetV1/Conv2d_4_depthwise/BatchNorm/beta/Read/ReadVariableOpAMobilenetV1/Conv2d_9_pointwise/BatchNorm/beta/Read/ReadVariableOpBMobilenetV1/Conv2d_5_pointwise/BatchNorm/gamma/Read/ReadVariableOpBMobilenetV1/Conv2d_2_depthwise/BatchNorm/gamma/Read/ReadVariableOpBMobilenetV1/Conv2d_4_pointwise/BatchNorm/gamma/Read/ReadVariableOpDMobilenetV1/Conv2d_6_depthwise/depthwise_weights/Read/ReadVariableOpBMobilenetV1/Conv2d_9_depthwise/BatchNorm/gamma/Read/ReadVariableOpMMobilenetV1/Conv2d_11_depthwise/BatchNorm/moving_variance/Read/ReadVariableOpCMobilenetV1/Conv2d_13_depthwise/BatchNorm/gamma/Read/ReadVariableOpMMobilenetV1/Conv2d_13_pointwise/BatchNorm/moving_variance/Read/ReadVariableOpIMobilenetV1/Conv2d_11_depthwise/BatchNorm/moving_mean/Read/ReadVariableOp:MobilenetV1/Conv2d_8_pointwise/weights/Read/ReadVariableOpHMobilenetV1/Conv2d_3_depthwise/BatchNorm/moving_mean/Read/ReadVariableOp:MobilenetV1/Conv2d_4_pointwise/weights/Read/ReadVariableOpLMobilenetV1/Conv2d_4_pointwise/BatchNorm/moving_variance/Read/ReadVariableOpAMobilenetV1/Conv2d_6_depthwise/BatchNorm/beta/Read/ReadVariableOpLMobilenetV1/Conv2d_6_pointwise/BatchNorm/moving_variance/Read/ReadVariableOpHMobilenetV1/Conv2d_9_depthwise/BatchNorm/moving_mean/Read/ReadVariableOpBMobilenetV1/Conv2d_9_pointwise/BatchNorm/gamma/Read/ReadVariableOpIMobilenetV1/Conv2d_12_pointwise/BatchNorm/moving_mean/Read/ReadVariableOpBMobilenetV1/Conv2d_13_pointwise/BatchNorm/beta/Read/ReadVariableOpHMobilenetV1/Conv2d_7_pointwise/BatchNorm/moving_mean/Read/ReadVariableOpIMobilenetV1/Conv2d_12_depthwise/BatchNorm/moving_mean/Read/ReadVariableOpDMobilenetV1/Conv2d_3_depthwise/depthwise_weights/Read/ReadVariableOpAMobilenetV1/Conv2d_4_pointwise/BatchNorm/beta/Read/ReadVariableOpHMobilenetV1/Conv2d_2_depthwise/BatchNorm/moving_mean/Read/ReadVariableOpAMobilenetV1/Conv2d_2_pointwise/BatchNorm/beta/Read/ReadVariableOpCMobilenetV1/Conv2d_10_depthwise/BatchNorm/gamma/Read/ReadVariableOpEMobilenetV1/Conv2d_12_depthwise/depthwise_weights/Read/ReadVariableOp7MobilenetV1/Conv2d_0/BatchNorm/beta/Read/ReadVariableOpEMobilenetV1/Conv2d_11_depthwise/depthwise_weights/Read/ReadVariableOpBMobilenetV1/Conv2d_11_depthwise/BatchNorm/beta/Read/ReadVariableOpMMobilenetV1/Conv2d_11_pointwise/BatchNorm/moving_variance/Read/ReadVariableOpHMobilenetV1/Conv2d_7_depthwise/BatchNorm/moving_mean/Read/ReadVariableOpBMobilenetV1/Conv2d_1_depthwise/BatchNorm/gamma/Read/ReadVariableOpLMobilenetV1/Conv2d_3_depthwise/BatchNorm/moving_variance/Read/ReadVariableOpHMobilenetV1/Conv2d_5_pointwise/BatchNorm/moving_mean/Read/ReadVariableOpCMobilenetV1/Conv2d_12_pointwise/BatchNorm/gamma/Read/ReadVariableOp:MobilenetV1/Conv2d_2_pointwise/weights/Read/ReadVariableOpLMobilenetV1/Conv2d_2_pointwise/BatchNorm/moving_variance/Read/ReadVariableOpBMobilenetV1/Conv2d_5_depthwise/BatchNorm/gamma/Read/ReadVariableOpEMobilenetV1/Conv2d_10_depthwise/depthwise_weights/Read/ReadVariableOpCMobilenetV1/Conv2d_11_pointwise/BatchNorm/gamma/Read/ReadVariableOpBMobilenetV1/Conv2d_6_pointwise/BatchNorm/gamma/Read/ReadVariableOpConst*?
Tin?
?2?	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *&
f!R
__inference__traced_save_7784
?9
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamesave_counter8MobilenetV1/Conv2d_1_depthwise/BatchNorm/moving_variance.MobilenetV1/Conv2d_3_depthwise/BatchNorm/gamma.MobilenetV1/Conv2d_7_pointwise/BatchNorm/gamma-MobilenetV1/Conv2d_8_pointwise/BatchNorm/beta.MobilenetV1/Conv2d_11_pointwise/BatchNorm/beta0MobilenetV1/Conv2d_4_depthwise/depthwise_weights(MobilenetV1/Logits/Conv2d_1c_1x1/weights*MobilenetV1/Conv2d_0/BatchNorm/moving_mean4MobilenetV1/Conv2d_3_pointwise/BatchNorm/moving_mean4MobilenetV1/Conv2d_4_depthwise/BatchNorm/moving_mean8MobilenetV1/Conv2d_5_depthwise/BatchNorm/moving_variance.MobilenetV1/Conv2d_6_depthwise/BatchNorm/gamma8MobilenetV1/Conv2d_7_pointwise/BatchNorm/moving_variance8MobilenetV1/Conv2d_9_depthwise/BatchNorm/moving_variance8MobilenetV1/Conv2d_9_pointwise/BatchNorm/moving_variance'MobilenetV1/Conv2d_11_pointwise/weights4MobilenetV1/Conv2d_1_pointwise/BatchNorm/moving_mean4MobilenetV1/Conv2d_5_depthwise/BatchNorm/moving_mean5MobilenetV1/Conv2d_10_pointwise/BatchNorm/moving_mean-MobilenetV1/Conv2d_7_pointwise/BatchNorm/betaMobilenetV1/Conv2d_0/weights8MobilenetV1/Conv2d_2_depthwise/BatchNorm/moving_variance8MobilenetV1/Conv2d_4_depthwise/BatchNorm/moving_variance4MobilenetV1/Conv2d_6_pointwise/BatchNorm/moving_mean-MobilenetV1/Conv2d_9_depthwise/BatchNorm/beta/MobilenetV1/Conv2d_11_depthwise/BatchNorm/gamma&MobilenetV1/Conv2d_6_pointwise/weights-MobilenetV1/Conv2d_6_pointwise/BatchNorm/beta-MobilenetV1/Conv2d_1_depthwise/BatchNorm/beta.MobilenetV1/Conv2d_3_pointwise/BatchNorm/gamma.MobilenetV1/Conv2d_10_depthwise/BatchNorm/beta'MobilenetV1/Logits/Conv2d_1c_1x1/biases4MobilenetV1/Conv2d_8_pointwise/BatchNorm/moving_mean-MobilenetV1/Conv2d_5_pointwise/BatchNorm/beta0MobilenetV1/Conv2d_2_depthwise/depthwise_weights&MobilenetV1/Conv2d_1_pointwise/weights.MobilenetV1/Conv2d_1_pointwise/BatchNorm/gamma-MobilenetV1/Conv2d_5_depthwise/BatchNorm/beta8MobilenetV1/Conv2d_5_pointwise/BatchNorm/moving_variance.MobilenetV1/Conv2d_8_depthwise/BatchNorm/gamma1MobilenetV1/Conv2d_13_depthwise/depthwise_weights8MobilenetV1/Conv2d_6_depthwise/BatchNorm/moving_variance&MobilenetV1/Conv2d_9_pointwise/weights9MobilenetV1/Conv2d_10_depthwise/BatchNorm/moving_variance.MobilenetV1/Conv2d_12_pointwise/BatchNorm/beta-MobilenetV1/Conv2d_2_depthwise/BatchNorm/beta4MobilenetV1/Conv2d_2_pointwise/BatchNorm/moving_mean-MobilenetV1/Conv2d_3_pointwise/BatchNorm/beta4MobilenetV1/Conv2d_6_depthwise/BatchNorm/moving_mean/MobilenetV1/Conv2d_10_pointwise/BatchNorm/gamma/MobilenetV1/Conv2d_12_depthwise/BatchNorm/gamma5MobilenetV1/Conv2d_13_depthwise/BatchNorm/moving_mean0MobilenetV1/Conv2d_5_depthwise/depthwise_weights.MobilenetV1/Conv2d_13_depthwise/BatchNorm/beta.MobilenetV1/Conv2d_8_pointwise/BatchNorm/gamma-MobilenetV1/Conv2d_3_depthwise/BatchNorm/beta.MobilenetV1/Conv2d_7_depthwise/BatchNorm/gamma8MobilenetV1/Conv2d_8_depthwise/BatchNorm/moving_variance5MobilenetV1/Conv2d_10_depthwise/BatchNorm/moving_mean9MobilenetV1/Conv2d_10_pointwise/BatchNorm/moving_variance'MobilenetV1/Conv2d_12_pointwise/weights9MobilenetV1/Conv2d_12_pointwise/BatchNorm/moving_variance.MobilenetV1/Conv2d_4_depthwise/BatchNorm/gamma4MobilenetV1/Conv2d_8_depthwise/BatchNorm/moving_mean5MobilenetV1/Conv2d_13_pointwise/BatchNorm/moving_mean8MobilenetV1/Conv2d_7_depthwise/BatchNorm/moving_variance&MobilenetV1/Conv2d_7_pointwise/weights&MobilenetV1/Conv2d_5_pointwise/weights8MobilenetV1/Conv2d_1_pointwise/BatchNorm/moving_variance4MobilenetV1/Conv2d_1_depthwise/BatchNorm/moving_mean8MobilenetV1/Conv2d_3_pointwise/BatchNorm/moving_variance4MobilenetV1/Conv2d_9_pointwise/BatchNorm/moving_mean.MobilenetV1/Conv2d_10_pointwise/BatchNorm/beta9MobilenetV1/Conv2d_12_depthwise/BatchNorm/moving_variance4MobilenetV1/Conv2d_4_pointwise/BatchNorm/moving_mean'MobilenetV1/Conv2d_10_pointwise/weights0MobilenetV1/Conv2d_1_depthwise/depthwise_weights-MobilenetV1/Conv2d_1_pointwise/BatchNorm/beta.MobilenetV1/Conv2d_0/BatchNorm/moving_variance.MobilenetV1/Conv2d_2_pointwise/BatchNorm/gamma0MobilenetV1/Conv2d_9_depthwise/depthwise_weights5MobilenetV1/Conv2d_11_pointwise/BatchNorm/moving_mean0MobilenetV1/Conv2d_8_depthwise/depthwise_weights-MobilenetV1/Conv2d_8_depthwise/BatchNorm/beta8MobilenetV1/Conv2d_8_pointwise/BatchNorm/moving_variance&MobilenetV1/Conv2d_3_pointwise/weights.MobilenetV1/Conv2d_12_depthwise/BatchNorm/beta9MobilenetV1/Conv2d_13_depthwise/BatchNorm/moving_variance'MobilenetV1/Conv2d_13_pointwise/weights0MobilenetV1/Conv2d_7_depthwise/depthwise_weights-MobilenetV1/Conv2d_7_depthwise/BatchNorm/beta/MobilenetV1/Conv2d_13_pointwise/BatchNorm/gamma$MobilenetV1/Conv2d_0/BatchNorm/gamma-MobilenetV1/Conv2d_4_depthwise/BatchNorm/beta-MobilenetV1/Conv2d_9_pointwise/BatchNorm/beta.MobilenetV1/Conv2d_5_pointwise/BatchNorm/gamma.MobilenetV1/Conv2d_2_depthwise/BatchNorm/gamma.MobilenetV1/Conv2d_4_pointwise/BatchNorm/gamma0MobilenetV1/Conv2d_6_depthwise/depthwise_weights.MobilenetV1/Conv2d_9_depthwise/BatchNorm/gamma9MobilenetV1/Conv2d_11_depthwise/BatchNorm/moving_variance/MobilenetV1/Conv2d_13_depthwise/BatchNorm/gamma9MobilenetV1/Conv2d_13_pointwise/BatchNorm/moving_variance5MobilenetV1/Conv2d_11_depthwise/BatchNorm/moving_mean&MobilenetV1/Conv2d_8_pointwise/weights4MobilenetV1/Conv2d_3_depthwise/BatchNorm/moving_mean&MobilenetV1/Conv2d_4_pointwise/weights8MobilenetV1/Conv2d_4_pointwise/BatchNorm/moving_variance-MobilenetV1/Conv2d_6_depthwise/BatchNorm/beta8MobilenetV1/Conv2d_6_pointwise/BatchNorm/moving_variance4MobilenetV1/Conv2d_9_depthwise/BatchNorm/moving_mean.MobilenetV1/Conv2d_9_pointwise/BatchNorm/gamma5MobilenetV1/Conv2d_12_pointwise/BatchNorm/moving_mean.MobilenetV1/Conv2d_13_pointwise/BatchNorm/beta4MobilenetV1/Conv2d_7_pointwise/BatchNorm/moving_mean5MobilenetV1/Conv2d_12_depthwise/BatchNorm/moving_mean0MobilenetV1/Conv2d_3_depthwise/depthwise_weights-MobilenetV1/Conv2d_4_pointwise/BatchNorm/beta4MobilenetV1/Conv2d_2_depthwise/BatchNorm/moving_mean-MobilenetV1/Conv2d_2_pointwise/BatchNorm/beta/MobilenetV1/Conv2d_10_depthwise/BatchNorm/gamma1MobilenetV1/Conv2d_12_depthwise/depthwise_weights#MobilenetV1/Conv2d_0/BatchNorm/beta1MobilenetV1/Conv2d_11_depthwise/depthwise_weights.MobilenetV1/Conv2d_11_depthwise/BatchNorm/beta9MobilenetV1/Conv2d_11_pointwise/BatchNorm/moving_variance4MobilenetV1/Conv2d_7_depthwise/BatchNorm/moving_mean.MobilenetV1/Conv2d_1_depthwise/BatchNorm/gamma8MobilenetV1/Conv2d_3_depthwise/BatchNorm/moving_variance4MobilenetV1/Conv2d_5_pointwise/BatchNorm/moving_mean/MobilenetV1/Conv2d_12_pointwise/BatchNorm/gamma&MobilenetV1/Conv2d_2_pointwise/weights8MobilenetV1/Conv2d_2_pointwise/BatchNorm/moving_variance.MobilenetV1/Conv2d_5_depthwise/BatchNorm/gamma1MobilenetV1/Conv2d_10_depthwise/depthwise_weights/MobilenetV1/Conv2d_11_pointwise/BatchNorm/gamma.MobilenetV1/Conv2d_6_pointwise/BatchNorm/gamma*?
Tin?
?2?*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__traced_restore_8208??"
?B
? 
#__inference_serving_default_fn_5484

inputs!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:#
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:#
	unknown_9:0

unknown_10:0

unknown_11:0

unknown_12:0

unknown_13:0$

unknown_14:0

unknown_15:0

unknown_16:0

unknown_17:0

unknown_18:0$

unknown_19:0`

unknown_20:`

unknown_21:`

unknown_22:`

unknown_23:`$

unknown_24:`

unknown_25:`

unknown_26:`

unknown_27:`

unknown_28:`$

unknown_29:``

unknown_30:`

unknown_31:`

unknown_32:`

unknown_33:`$

unknown_34:`

unknown_35:`

unknown_36:`

unknown_37:`

unknown_38:`%

unknown_39:`?

unknown_40:	?

unknown_41:	?

unknown_42:	?

unknown_43:	?%

unknown_44:?

unknown_45:	?

unknown_46:	?

unknown_47:	?

unknown_48:	?&

unknown_49:??

unknown_50:	?

unknown_51:	?

unknown_52:	?

unknown_53:	?%

unknown_54:?

unknown_55:	?

unknown_56:	?

unknown_57:	?

unknown_58:	?&

unknown_59:??

unknown_60:	?

unknown_61:	?

unknown_62:	?

unknown_63:	?%

unknown_64:?

unknown_65:	?

unknown_66:	?

unknown_67:	?

unknown_68:	?&

unknown_69:??

unknown_70:	?

unknown_71:	?

unknown_72:	?

unknown_73:	?%

unknown_74:?

unknown_75:	?

unknown_76:	?

unknown_77:	?

unknown_78:	?&

unknown_79:??

unknown_80:	?

unknown_81:	?

unknown_82:	?

unknown_83:	?%

unknown_84:?

unknown_85:	?

unknown_86:	?

unknown_87:	?

unknown_88:	?&

unknown_89:??

unknown_90:	?

unknown_91:	?

unknown_92:	?

unknown_93:	?%

unknown_94:?

unknown_95:	?

unknown_96:	?

unknown_97:	?

unknown_98:	?&

unknown_99:??
unknown_100:	?
unknown_101:	?
unknown_102:	?
unknown_103:	?&
unknown_104:?
unknown_105:	?
unknown_106:	?
unknown_107:	?
unknown_108:	?'
unknown_109:??
unknown_110:	?
unknown_111:	?
unknown_112:	?
unknown_113:	?&
unknown_114:?
unknown_115:	?
unknown_116:	?
unknown_117:	?
unknown_118:	?'
unknown_119:??
unknown_120:	?
unknown_121:	?
unknown_122:	?
unknown_123:	?&
unknown_124:?
unknown_125:	?
unknown_126:	?
unknown_127:	?
unknown_128:	?'
unknown_129:??
unknown_130:	?
unknown_131:	?
unknown_132:	?
unknown_133:	?'
unknown_134:??
unknown_135:	?
identity??StatefulPartitionedCallS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2
Const?
StatefulPartitionedCallStatefulPartitionedCallinputsConst:output:0unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58
unknown_59
unknown_60
unknown_61
unknown_62
unknown_63
unknown_64
unknown_65
unknown_66
unknown_67
unknown_68
unknown_69
unknown_70
unknown_71
unknown_72
unknown_73
unknown_74
unknown_75
unknown_76
unknown_77
unknown_78
unknown_79
unknown_80
unknown_81
unknown_82
unknown_83
unknown_84
unknown_85
unknown_86
unknown_87
unknown_88
unknown_89
unknown_90
unknown_91
unknown_92
unknown_93
unknown_94
unknown_95
unknown_96
unknown_97
unknown_98
unknown_99unknown_100unknown_101unknown_102unknown_103unknown_104unknown_105unknown_106unknown_107unknown_108unknown_109unknown_110unknown_111unknown_112unknown_113unknown_114unknown_115unknown_116unknown_117unknown_118unknown_119unknown_120unknown_121unknown_122unknown_123unknown_124unknown_125unknown_126unknown_127unknown_128unknown_129unknown_130unknown_131unknown_132unknown_133unknown_134unknown_135*?
Tin?
?2?*+
Tout#
!2*?

_output_shapes?

?
:??????????:+???????????????????????????:,????????????????????????????:,????????????????????????????:,????????????????????????????:,????????????????????????????:,????????????????????????????:,????????????????????????????:,????????????????????????????:,????????????????????????????:+???????????????????????????:+???????????????????????????0:+???????????????????????????0:+???????????????????????????`:+???????????????????????????`:+???????????????????????????`:+???????????????????????????`:,????????????????????????????:,????????????????????????????:,????????????????????????????:,????????????????????????????:,????????????????????????????:,????????????????????????????:,????????????????????????????:,????????????????????????????:,????????????????????????????:,????????????????????????????:,????????????????????????????:??????????:??????????:??????????*?
_read_only_resource_inputs?
??	
 !"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~???????????*-
config_proto

CPU

GPU 2J 8? * 
fR
__inference_pruned_33932
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

IdentityD
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:+???????????????????????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?t
?$
__inference_call_fn_6758

inputs
batch_norm_momentum!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:#
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:#
	unknown_9:0

unknown_10:0

unknown_11:0

unknown_12:0

unknown_13:0$

unknown_14:0

unknown_15:0

unknown_16:0

unknown_17:0

unknown_18:0$

unknown_19:0`

unknown_20:`

unknown_21:`

unknown_22:`

unknown_23:`$

unknown_24:`

unknown_25:`

unknown_26:`

unknown_27:`

unknown_28:`$

unknown_29:``

unknown_30:`

unknown_31:`

unknown_32:`

unknown_33:`$

unknown_34:`

unknown_35:`

unknown_36:`

unknown_37:`

unknown_38:`%

unknown_39:`?

unknown_40:	?

unknown_41:	?

unknown_42:	?

unknown_43:	?%

unknown_44:?

unknown_45:	?

unknown_46:	?

unknown_47:	?

unknown_48:	?&

unknown_49:??

unknown_50:	?

unknown_51:	?

unknown_52:	?

unknown_53:	?%

unknown_54:?

unknown_55:	?

unknown_56:	?

unknown_57:	?

unknown_58:	?&

unknown_59:??

unknown_60:	?

unknown_61:	?

unknown_62:	?

unknown_63:	?%

unknown_64:?

unknown_65:	?

unknown_66:	?

unknown_67:	?

unknown_68:	?&

unknown_69:??

unknown_70:	?

unknown_71:	?

unknown_72:	?

unknown_73:	?%

unknown_74:?

unknown_75:	?

unknown_76:	?

unknown_77:	?

unknown_78:	?&

unknown_79:??

unknown_80:	?

unknown_81:	?

unknown_82:	?

unknown_83:	?%

unknown_84:?

unknown_85:	?

unknown_86:	?

unknown_87:	?

unknown_88:	?&

unknown_89:??

unknown_90:	?

unknown_91:	?

unknown_92:	?

unknown_93:	?%

unknown_94:?

unknown_95:	?

unknown_96:	?

unknown_97:	?

unknown_98:	?&

unknown_99:??
unknown_100:	?
unknown_101:	?
unknown_102:	?
unknown_103:	?&
unknown_104:?
unknown_105:	?
unknown_106:	?
unknown_107:	?
unknown_108:	?'
unknown_109:??
unknown_110:	?
unknown_111:	?
unknown_112:	?
unknown_113:	?&
unknown_114:?
unknown_115:	?
unknown_116:	?
unknown_117:	?
unknown_118:	?'
unknown_119:??
unknown_120:	?
unknown_121:	?
unknown_122:	?
unknown_123:	?&
unknown_124:?
unknown_125:	?
unknown_126:	?
unknown_127:	?
unknown_128:	?'
unknown_129:??
unknown_130:	?
unknown_131:	?
unknown_132:	?
unknown_133:	?'
unknown_134:??
unknown_135:	?
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6

identity_7

identity_8

identity_9
identity_10
identity_11
identity_12
identity_13
identity_14
identity_15
identity_16
identity_17
identity_18
identity_19
identity_20
identity_21
identity_22
identity_23
identity_24
identity_25
identity_26
identity_27
identity_28
identity_29
identity_30??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_norm_momentumunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58
unknown_59
unknown_60
unknown_61
unknown_62
unknown_63
unknown_64
unknown_65
unknown_66
unknown_67
unknown_68
unknown_69
unknown_70
unknown_71
unknown_72
unknown_73
unknown_74
unknown_75
unknown_76
unknown_77
unknown_78
unknown_79
unknown_80
unknown_81
unknown_82
unknown_83
unknown_84
unknown_85
unknown_86
unknown_87
unknown_88
unknown_89
unknown_90
unknown_91
unknown_92
unknown_93
unknown_94
unknown_95
unknown_96
unknown_97
unknown_98
unknown_99unknown_100unknown_101unknown_102unknown_103unknown_104unknown_105unknown_106unknown_107unknown_108unknown_109unknown_110unknown_111unknown_112unknown_113unknown_114unknown_115unknown_116unknown_117unknown_118unknown_119unknown_120unknown_121unknown_122unknown_123unknown_124unknown_125unknown_126unknown_127unknown_128unknown_129unknown_130unknown_131unknown_132unknown_133unknown_134unknown_135*?
Tin?
?2?*+
Tout#
!2*?

_output_shapes?

?
:??????????:+???????????????????????????:,????????????????????????????:,????????????????????????????:,????????????????????????????:,????????????????????????????:,????????????????????????????:,????????????????????????????:,????????????????????????????:,????????????????????????????:+???????????????????????????:+???????????????????????????0:+???????????????????????????0:+???????????????????????????`:+???????????????????????????`:+???????????????????????????`:+???????????????????????????`:,????????????????????????????:,????????????????????????????:,????????????????????????????:,????????????????????????????:,????????????????????????????:,????????????????????????????:,????????????????????????????:,????????????????????????????:,????????????????????????????:,????????????????????????????:,????????????????????????????:??????????:??????????:??????????*?
_read_only_resource_inputs?
??	
 !"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~???????????*-
config_proto

CPU

GPU 2J 8? * 
fR
__inference_pruned_33932
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:1^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:2^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:3^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity_2?

Identity_3Identity StatefulPartitionedCall:output:4^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity_3?

Identity_4Identity StatefulPartitionedCall:output:5^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity_4?

Identity_5Identity StatefulPartitionedCall:output:6^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity_5?

Identity_6Identity StatefulPartitionedCall:output:7^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity_6?

Identity_7Identity StatefulPartitionedCall:output:8^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity_7?

Identity_8Identity StatefulPartitionedCall:output:9^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity_8?

Identity_9Identity!StatefulPartitionedCall:output:10^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity_9?
Identity_10Identity!StatefulPartitionedCall:output:11^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????02
Identity_10?
Identity_11Identity!StatefulPartitionedCall:output:12^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????02
Identity_11?
Identity_12Identity!StatefulPartitionedCall:output:13^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????`2
Identity_12?
Identity_13Identity!StatefulPartitionedCall:output:14^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????`2
Identity_13?
Identity_14Identity!StatefulPartitionedCall:output:15^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????`2
Identity_14?
Identity_15Identity!StatefulPartitionedCall:output:16^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????`2
Identity_15?
Identity_16Identity!StatefulPartitionedCall:output:17^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????2
Identity_16?
Identity_17Identity!StatefulPartitionedCall:output:18^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????2
Identity_17?
Identity_18Identity!StatefulPartitionedCall:output:19^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????2
Identity_18?
Identity_19Identity!StatefulPartitionedCall:output:20^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????2
Identity_19?
Identity_20Identity!StatefulPartitionedCall:output:21^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????2
Identity_20?
Identity_21Identity!StatefulPartitionedCall:output:22^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????2
Identity_21?
Identity_22Identity!StatefulPartitionedCall:output:23^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????2
Identity_22?
Identity_23Identity!StatefulPartitionedCall:output:24^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????2
Identity_23?
Identity_24Identity!StatefulPartitionedCall:output:25^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????2
Identity_24?
Identity_25Identity!StatefulPartitionedCall:output:26^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????2
Identity_25?
Identity_26Identity!StatefulPartitionedCall:output:27^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????2
Identity_26?
Identity_27Identity!StatefulPartitionedCall:output:28^NoOp*
T0*(
_output_shapes
:??????????2
Identity_27?
Identity_28Identity!StatefulPartitionedCall:output:29^NoOp*
T0*(
_output_shapes
:??????????2
Identity_28?
Identity_29Identity!StatefulPartitionedCall:output:30^NoOp*
T0*0
_output_shapes
:??????????2
Identity_29?
Identity_30Identity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2
Identity_30D
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"#
identity_10Identity_10:output:0"#
identity_11Identity_11:output:0"#
identity_12Identity_12:output:0"#
identity_13Identity_13:output:0"#
identity_14Identity_14:output:0"#
identity_15Identity_15:output:0"#
identity_16Identity_16:output:0"#
identity_17Identity_17:output:0"#
identity_18Identity_18:output:0"#
identity_19Identity_19:output:0"!

identity_2Identity_2:output:0"#
identity_20Identity_20:output:0"#
identity_21Identity_21:output:0"#
identity_22Identity_22:output:0"#
identity_23Identity_23:output:0"#
identity_24Identity_24:output:0"#
identity_25Identity_25:output:0"#
identity_26Identity_26:output:0"#
identity_27Identity_27:output:0"#
identity_28Identity_28:output:0"#
identity_29Identity_29:output:0"!

identity_3Identity_3:output:0"#
identity_30Identity_30:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0"!

identity_7Identity_7:output:0"!

identity_8Identity_8:output:0"!

identity_9Identity_9:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:+???????????????????????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs:KG

_output_shapes
: 
-
_user_specified_namebatch_norm_momentum
?s
?$
__inference_call_fn_6108

inputs
batch_norm_momentum!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:#
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:#
	unknown_9:0

unknown_10:0

unknown_11:0

unknown_12:0

unknown_13:0$

unknown_14:0

unknown_15:0

unknown_16:0

unknown_17:0

unknown_18:0$

unknown_19:0`

unknown_20:`

unknown_21:`

unknown_22:`

unknown_23:`$

unknown_24:`

unknown_25:`

unknown_26:`

unknown_27:`

unknown_28:`$

unknown_29:``

unknown_30:`

unknown_31:`

unknown_32:`

unknown_33:`$

unknown_34:`

unknown_35:`

unknown_36:`

unknown_37:`

unknown_38:`%

unknown_39:`?

unknown_40:	?

unknown_41:	?

unknown_42:	?

unknown_43:	?%

unknown_44:?

unknown_45:	?

unknown_46:	?

unknown_47:	?

unknown_48:	?&

unknown_49:??

unknown_50:	?

unknown_51:	?

unknown_52:	?

unknown_53:	?%

unknown_54:?

unknown_55:	?

unknown_56:	?

unknown_57:	?

unknown_58:	?&

unknown_59:??

unknown_60:	?

unknown_61:	?

unknown_62:	?

unknown_63:	?%

unknown_64:?

unknown_65:	?

unknown_66:	?

unknown_67:	?

unknown_68:	?&

unknown_69:??

unknown_70:	?

unknown_71:	?

unknown_72:	?

unknown_73:	?%

unknown_74:?

unknown_75:	?

unknown_76:	?

unknown_77:	?

unknown_78:	?&

unknown_79:??

unknown_80:	?

unknown_81:	?

unknown_82:	?

unknown_83:	?%

unknown_84:?

unknown_85:	?

unknown_86:	?

unknown_87:	?

unknown_88:	?&

unknown_89:??

unknown_90:	?

unknown_91:	?

unknown_92:	?

unknown_93:	?%

unknown_94:?

unknown_95:	?

unknown_96:	?

unknown_97:	?

unknown_98:	?&

unknown_99:??
unknown_100:	?
unknown_101:	?
unknown_102:	?
unknown_103:	?&
unknown_104:?
unknown_105:	?
unknown_106:	?
unknown_107:	?
unknown_108:	?'
unknown_109:??
unknown_110:	?
unknown_111:	?
unknown_112:	?
unknown_113:	?&
unknown_114:?
unknown_115:	?
unknown_116:	?
unknown_117:	?
unknown_118:	?'
unknown_119:??
unknown_120:	?
unknown_121:	?
unknown_122:	?
unknown_123:	?&
unknown_124:?
unknown_125:	?
unknown_126:	?
unknown_127:	?
unknown_128:	?'
unknown_129:??
unknown_130:	?
unknown_131:	?
unknown_132:	?
unknown_133:	?'
unknown_134:??
unknown_135:	?
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6

identity_7

identity_8

identity_9
identity_10
identity_11
identity_12
identity_13
identity_14
identity_15
identity_16
identity_17
identity_18
identity_19
identity_20
identity_21
identity_22
identity_23
identity_24
identity_25
identity_26
identity_27
identity_28
identity_29
identity_30??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_norm_momentumunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58
unknown_59
unknown_60
unknown_61
unknown_62
unknown_63
unknown_64
unknown_65
unknown_66
unknown_67
unknown_68
unknown_69
unknown_70
unknown_71
unknown_72
unknown_73
unknown_74
unknown_75
unknown_76
unknown_77
unknown_78
unknown_79
unknown_80
unknown_81
unknown_82
unknown_83
unknown_84
unknown_85
unknown_86
unknown_87
unknown_88
unknown_89
unknown_90
unknown_91
unknown_92
unknown_93
unknown_94
unknown_95
unknown_96
unknown_97
unknown_98
unknown_99unknown_100unknown_101unknown_102unknown_103unknown_104unknown_105unknown_106unknown_107unknown_108unknown_109unknown_110unknown_111unknown_112unknown_113unknown_114unknown_115unknown_116unknown_117unknown_118unknown_119unknown_120unknown_121unknown_122unknown_123unknown_124unknown_125unknown_126unknown_127unknown_128unknown_129unknown_130unknown_131unknown_132unknown_133unknown_134unknown_135*?
Tin?
?2?*+
Tout#
!2*?

_output_shapes?

?
:??????????:+???????????????????????????:,????????????????????????????:,????????????????????????????:,????????????????????????????:,????????????????????????????:,????????????????????????????:,????????????????????????????:,????????????????????????????:,????????????????????????????:+???????????????????????????:+???????????????????????????0:+???????????????????????????0:+???????????????????????????`:+???????????????????????????`:+???????????????????????????`:+???????????????????????????`:,????????????????????????????:,????????????????????????????:,????????????????????????????:,????????????????????????????:,????????????????????????????:,????????????????????????????:,????????????????????????????:,????????????????????????????:,????????????????????????????:,????????????????????????????:,????????????????????????????:??????????:??????????:??????????*|
_read_only_resource_inputs^
\Z	 !"%&'*+,/014569:;>?@CDEHIJMNORSTWXY\]^abcfghklmpqruvwz{|???????*-
config_proto

CPU

GPU 2J 8? * 
fR
__inference_pruned_28842
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:1^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:2^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:3^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity_2?

Identity_3Identity StatefulPartitionedCall:output:4^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity_3?

Identity_4Identity StatefulPartitionedCall:output:5^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity_4?

Identity_5Identity StatefulPartitionedCall:output:6^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity_5?

Identity_6Identity StatefulPartitionedCall:output:7^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity_6?

Identity_7Identity StatefulPartitionedCall:output:8^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity_7?

Identity_8Identity StatefulPartitionedCall:output:9^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity_8?

Identity_9Identity!StatefulPartitionedCall:output:10^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity_9?
Identity_10Identity!StatefulPartitionedCall:output:11^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????02
Identity_10?
Identity_11Identity!StatefulPartitionedCall:output:12^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????02
Identity_11?
Identity_12Identity!StatefulPartitionedCall:output:13^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????`2
Identity_12?
Identity_13Identity!StatefulPartitionedCall:output:14^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????`2
Identity_13?
Identity_14Identity!StatefulPartitionedCall:output:15^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????`2
Identity_14?
Identity_15Identity!StatefulPartitionedCall:output:16^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????`2
Identity_15?
Identity_16Identity!StatefulPartitionedCall:output:17^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????2
Identity_16?
Identity_17Identity!StatefulPartitionedCall:output:18^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????2
Identity_17?
Identity_18Identity!StatefulPartitionedCall:output:19^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????2
Identity_18?
Identity_19Identity!StatefulPartitionedCall:output:20^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????2
Identity_19?
Identity_20Identity!StatefulPartitionedCall:output:21^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????2
Identity_20?
Identity_21Identity!StatefulPartitionedCall:output:22^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????2
Identity_21?
Identity_22Identity!StatefulPartitionedCall:output:23^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????2
Identity_22?
Identity_23Identity!StatefulPartitionedCall:output:24^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????2
Identity_23?
Identity_24Identity!StatefulPartitionedCall:output:25^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????2
Identity_24?
Identity_25Identity!StatefulPartitionedCall:output:26^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????2
Identity_25?
Identity_26Identity!StatefulPartitionedCall:output:27^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????2
Identity_26?
Identity_27Identity!StatefulPartitionedCall:output:28^NoOp*
T0*(
_output_shapes
:??????????2
Identity_27?
Identity_28Identity!StatefulPartitionedCall:output:29^NoOp*
T0*(
_output_shapes
:??????????2
Identity_28?
Identity_29Identity!StatefulPartitionedCall:output:30^NoOp*
T0*0
_output_shapes
:??????????2
Identity_29?
Identity_30Identity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2
Identity_30D
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"#
identity_10Identity_10:output:0"#
identity_11Identity_11:output:0"#
identity_12Identity_12:output:0"#
identity_13Identity_13:output:0"#
identity_14Identity_14:output:0"#
identity_15Identity_15:output:0"#
identity_16Identity_16:output:0"#
identity_17Identity_17:output:0"#
identity_18Identity_18:output:0"#
identity_19Identity_19:output:0"!

identity_2Identity_2:output:0"#
identity_20Identity_20:output:0"#
identity_21Identity_21:output:0"#
identity_22Identity_22:output:0"#
identity_23Identity_23:output:0"#
identity_24Identity_24:output:0"#
identity_25Identity_25:output:0"#
identity_26Identity_26:output:0"#
identity_27Identity_27:output:0"#
identity_28Identity_28:output:0"#
identity_29Identity_29:output:0"!

identity_3Identity_3:output:0"#
identity_30Identity_30:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0"!

identity_7Identity_7:output:0"!

identity_8Identity_8:output:0"!

identity_9Identity_9:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:+???????????????????????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs:KG

_output_shapes
: 
-
_user_specified_namebatch_norm_momentum
?5
? 
__inference_<lambda>_7347!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:#
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:#
	unknown_9:0

unknown_10:0

unknown_11:0

unknown_12:0

unknown_13:0$

unknown_14:0

unknown_15:0

unknown_16:0

unknown_17:0

unknown_18:0$

unknown_19:0`

unknown_20:`

unknown_21:`

unknown_22:`

unknown_23:`$

unknown_24:`

unknown_25:`

unknown_26:`

unknown_27:`

unknown_28:`$

unknown_29:``

unknown_30:`

unknown_31:`

unknown_32:`

unknown_33:`$

unknown_34:`

unknown_35:`

unknown_36:`

unknown_37:`

unknown_38:`%

unknown_39:`?

unknown_40:	?

unknown_41:	?

unknown_42:	?

unknown_43:	?%

unknown_44:?

unknown_45:	?

unknown_46:	?

unknown_47:	?

unknown_48:	?&

unknown_49:??

unknown_50:	?

unknown_51:	?

unknown_52:	?

unknown_53:	?%

unknown_54:?

unknown_55:	?

unknown_56:	?

unknown_57:	?

unknown_58:	?&

unknown_59:??

unknown_60:	?

unknown_61:	?

unknown_62:	?

unknown_63:	?%

unknown_64:?

unknown_65:	?

unknown_66:	?

unknown_67:	?

unknown_68:	?&

unknown_69:??

unknown_70:	?

unknown_71:	?

unknown_72:	?

unknown_73:	?%

unknown_74:?

unknown_75:	?

unknown_76:	?

unknown_77:	?

unknown_78:	?&

unknown_79:??

unknown_80:	?

unknown_81:	?

unknown_82:	?

unknown_83:	?%

unknown_84:?

unknown_85:	?

unknown_86:	?

unknown_87:	?

unknown_88:	?&

unknown_89:??

unknown_90:	?

unknown_91:	?

unknown_92:	?

unknown_93:	?%

unknown_94:?

unknown_95:	?

unknown_96:	?

unknown_97:	?

unknown_98:	?&

unknown_99:??
unknown_100:	?
unknown_101:	?
unknown_102:	?
unknown_103:	?&
unknown_104:?
unknown_105:	?
unknown_106:	?
unknown_107:	?
unknown_108:	?'
unknown_109:??
unknown_110:	?
unknown_111:	?
unknown_112:	?
unknown_113:	?&
unknown_114:?
unknown_115:	?
unknown_116:	?
unknown_117:	?
unknown_118:	?'
unknown_119:??
unknown_120:	?
unknown_121:	?
unknown_122:	?
unknown_123:	?&
unknown_124:?
unknown_125:	?
unknown_126:	?
unknown_127:	?
unknown_128:	?'
unknown_129:??
unknown_130:	?
unknown_131:	?
unknown_132:	?
unknown_133:	?'
unknown_134:??
unknown_135:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58
unknown_59
unknown_60
unknown_61
unknown_62
unknown_63
unknown_64
unknown_65
unknown_66
unknown_67
unknown_68
unknown_69
unknown_70
unknown_71
unknown_72
unknown_73
unknown_74
unknown_75
unknown_76
unknown_77
unknown_78
unknown_79
unknown_80
unknown_81
unknown_82
unknown_83
unknown_84
unknown_85
unknown_86
unknown_87
unknown_88
unknown_89
unknown_90
unknown_91
unknown_92
unknown_93
unknown_94
unknown_95
unknown_96
unknown_97
unknown_98
unknown_99unknown_100unknown_101unknown_102unknown_103unknown_104unknown_105unknown_106unknown_107unknown_108unknown_109unknown_110unknown_111unknown_112unknown_113unknown_114unknown_115unknown_116unknown_117unknown_118unknown_119unknown_120unknown_121unknown_122unknown_123unknown_124unknown_125unknown_126unknown_127unknown_128unknown_129unknown_130unknown_131unknown_132unknown_133unknown_134unknown_135*?
Tin?
?2?*
Tout
2*
_output_shapes
: *?
_read_only_resource_inputs?
?? 	
 !"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~?????????*-
config_proto

CPU

GPU 2J 8? * 
fR
__inference_pruned_35932
StatefulPartitionedCallj
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes
: 2

IdentityD
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall
?B
? 
__inference_call_fn_6418

inputs
batch_norm_momentum!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:#
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:#
	unknown_9:0

unknown_10:0

unknown_11:0

unknown_12:0

unknown_13:0$

unknown_14:0

unknown_15:0

unknown_16:0

unknown_17:0

unknown_18:0$

unknown_19:0`

unknown_20:`

unknown_21:`

unknown_22:`

unknown_23:`$

unknown_24:`

unknown_25:`

unknown_26:`

unknown_27:`

unknown_28:`$

unknown_29:``

unknown_30:`

unknown_31:`

unknown_32:`

unknown_33:`$

unknown_34:`

unknown_35:`

unknown_36:`

unknown_37:`

unknown_38:`%

unknown_39:`?

unknown_40:	?

unknown_41:	?

unknown_42:	?

unknown_43:	?%

unknown_44:?

unknown_45:	?

unknown_46:	?

unknown_47:	?

unknown_48:	?&

unknown_49:??

unknown_50:	?

unknown_51:	?

unknown_52:	?

unknown_53:	?%

unknown_54:?

unknown_55:	?

unknown_56:	?

unknown_57:	?

unknown_58:	?&

unknown_59:??

unknown_60:	?

unknown_61:	?

unknown_62:	?

unknown_63:	?%

unknown_64:?

unknown_65:	?

unknown_66:	?

unknown_67:	?

unknown_68:	?&

unknown_69:??

unknown_70:	?

unknown_71:	?

unknown_72:	?

unknown_73:	?%

unknown_74:?

unknown_75:	?

unknown_76:	?

unknown_77:	?

unknown_78:	?&

unknown_79:??

unknown_80:	?

unknown_81:	?

unknown_82:	?

unknown_83:	?%

unknown_84:?

unknown_85:	?

unknown_86:	?

unknown_87:	?

unknown_88:	?&

unknown_89:??

unknown_90:	?

unknown_91:	?

unknown_92:	?

unknown_93:	?%

unknown_94:?

unknown_95:	?

unknown_96:	?

unknown_97:	?

unknown_98:	?&

unknown_99:??
unknown_100:	?
unknown_101:	?
unknown_102:	?
unknown_103:	?&
unknown_104:?
unknown_105:	?
unknown_106:	?
unknown_107:	?
unknown_108:	?'
unknown_109:??
unknown_110:	?
unknown_111:	?
unknown_112:	?
unknown_113:	?&
unknown_114:?
unknown_115:	?
unknown_116:	?
unknown_117:	?
unknown_118:	?'
unknown_119:??
unknown_120:	?
unknown_121:	?
unknown_122:	?
unknown_123:	?&
unknown_124:?
unknown_125:	?
unknown_126:	?
unknown_127:	?
unknown_128:	?'
unknown_129:??
unknown_130:	?
unknown_131:	?
unknown_132:	?
unknown_133:	?'
unknown_134:??
unknown_135:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_norm_momentumunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58
unknown_59
unknown_60
unknown_61
unknown_62
unknown_63
unknown_64
unknown_65
unknown_66
unknown_67
unknown_68
unknown_69
unknown_70
unknown_71
unknown_72
unknown_73
unknown_74
unknown_75
unknown_76
unknown_77
unknown_78
unknown_79
unknown_80
unknown_81
unknown_82
unknown_83
unknown_84
unknown_85
unknown_86
unknown_87
unknown_88
unknown_89
unknown_90
unknown_91
unknown_92
unknown_93
unknown_94
unknown_95
unknown_96
unknown_97
unknown_98
unknown_99unknown_100unknown_101unknown_102unknown_103unknown_104unknown_105unknown_106unknown_107unknown_108unknown_109unknown_110unknown_111unknown_112unknown_113unknown_114unknown_115unknown_116unknown_117unknown_118unknown_119unknown_120unknown_121unknown_122unknown_123unknown_124unknown_125unknown_126unknown_127unknown_128unknown_129unknown_130unknown_131unknown_132unknown_133unknown_134unknown_135*?
Tin?
?2?*+
Tout#
!2*?

_output_shapes?

?
:??????????:+???????????????????????????:,????????????????????????????:,????????????????????????????:,????????????????????????????:,????????????????????????????:,????????????????????????????:,????????????????????????????:,????????????????????????????:,????????????????????????????:+???????????????????????????:+???????????????????????????0:+???????????????????????????0:+???????????????????????????`:+???????????????????????????`:+???????????????????????????`:+???????????????????????????`:,????????????????????????????:,????????????????????????????:,????????????????????????????:,????????????????????????????:,????????????????????????????:,????????????????????????????:,????????????????????????????:,????????????????????????????:,????????????????????????????:,????????????????????????????:,????????????????????????????:??????????:??????????:??????????*|
_read_only_resource_inputs^
\Z	 !"%&'*+,/014569:;>?@CDEHIJMNORSTWXY\]^abcfghklmpqruvwz{|???????*-
config_proto

CPU

GPU 2J 8? * 
fR
__inference_pruned_28842
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

IdentityD
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:+???????????????????????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs:KG

_output_shapes
: 
-
_user_specified_namebatch_norm_momentum
??
?X
__inference__traced_save_7784
file_prefix+
'savev2_save_counter_read_readvariableop	W
Ssavev2_mobilenetv1_conv2d_1_depthwise_batchnorm_moving_variance_read_readvariableopM
Isavev2_mobilenetv1_conv2d_3_depthwise_batchnorm_gamma_read_readvariableopM
Isavev2_mobilenetv1_conv2d_7_pointwise_batchnorm_gamma_read_readvariableopL
Hsavev2_mobilenetv1_conv2d_8_pointwise_batchnorm_beta_read_readvariableopM
Isavev2_mobilenetv1_conv2d_11_pointwise_batchnorm_beta_read_readvariableopO
Ksavev2_mobilenetv1_conv2d_4_depthwise_depthwise_weights_read_readvariableopG
Csavev2_mobilenetv1_logits_conv2d_1c_1x1_weights_read_readvariableopI
Esavev2_mobilenetv1_conv2d_0_batchnorm_moving_mean_read_readvariableopS
Osavev2_mobilenetv1_conv2d_3_pointwise_batchnorm_moving_mean_read_readvariableopS
Osavev2_mobilenetv1_conv2d_4_depthwise_batchnorm_moving_mean_read_readvariableopW
Ssavev2_mobilenetv1_conv2d_5_depthwise_batchnorm_moving_variance_read_readvariableopM
Isavev2_mobilenetv1_conv2d_6_depthwise_batchnorm_gamma_read_readvariableopW
Ssavev2_mobilenetv1_conv2d_7_pointwise_batchnorm_moving_variance_read_readvariableopW
Ssavev2_mobilenetv1_conv2d_9_depthwise_batchnorm_moving_variance_read_readvariableopW
Ssavev2_mobilenetv1_conv2d_9_pointwise_batchnorm_moving_variance_read_readvariableopF
Bsavev2_mobilenetv1_conv2d_11_pointwise_weights_read_readvariableopS
Osavev2_mobilenetv1_conv2d_1_pointwise_batchnorm_moving_mean_read_readvariableopS
Osavev2_mobilenetv1_conv2d_5_depthwise_batchnorm_moving_mean_read_readvariableopT
Psavev2_mobilenetv1_conv2d_10_pointwise_batchnorm_moving_mean_read_readvariableopL
Hsavev2_mobilenetv1_conv2d_7_pointwise_batchnorm_beta_read_readvariableop;
7savev2_mobilenetv1_conv2d_0_weights_read_readvariableopW
Ssavev2_mobilenetv1_conv2d_2_depthwise_batchnorm_moving_variance_read_readvariableopW
Ssavev2_mobilenetv1_conv2d_4_depthwise_batchnorm_moving_variance_read_readvariableopS
Osavev2_mobilenetv1_conv2d_6_pointwise_batchnorm_moving_mean_read_readvariableopL
Hsavev2_mobilenetv1_conv2d_9_depthwise_batchnorm_beta_read_readvariableopN
Jsavev2_mobilenetv1_conv2d_11_depthwise_batchnorm_gamma_read_readvariableopE
Asavev2_mobilenetv1_conv2d_6_pointwise_weights_read_readvariableopL
Hsavev2_mobilenetv1_conv2d_6_pointwise_batchnorm_beta_read_readvariableopL
Hsavev2_mobilenetv1_conv2d_1_depthwise_batchnorm_beta_read_readvariableopM
Isavev2_mobilenetv1_conv2d_3_pointwise_batchnorm_gamma_read_readvariableopM
Isavev2_mobilenetv1_conv2d_10_depthwise_batchnorm_beta_read_readvariableopF
Bsavev2_mobilenetv1_logits_conv2d_1c_1x1_biases_read_readvariableopS
Osavev2_mobilenetv1_conv2d_8_pointwise_batchnorm_moving_mean_read_readvariableopL
Hsavev2_mobilenetv1_conv2d_5_pointwise_batchnorm_beta_read_readvariableopO
Ksavev2_mobilenetv1_conv2d_2_depthwise_depthwise_weights_read_readvariableopE
Asavev2_mobilenetv1_conv2d_1_pointwise_weights_read_readvariableopM
Isavev2_mobilenetv1_conv2d_1_pointwise_batchnorm_gamma_read_readvariableopL
Hsavev2_mobilenetv1_conv2d_5_depthwise_batchnorm_beta_read_readvariableopW
Ssavev2_mobilenetv1_conv2d_5_pointwise_batchnorm_moving_variance_read_readvariableopM
Isavev2_mobilenetv1_conv2d_8_depthwise_batchnorm_gamma_read_readvariableopP
Lsavev2_mobilenetv1_conv2d_13_depthwise_depthwise_weights_read_readvariableopW
Ssavev2_mobilenetv1_conv2d_6_depthwise_batchnorm_moving_variance_read_readvariableopE
Asavev2_mobilenetv1_conv2d_9_pointwise_weights_read_readvariableopX
Tsavev2_mobilenetv1_conv2d_10_depthwise_batchnorm_moving_variance_read_readvariableopM
Isavev2_mobilenetv1_conv2d_12_pointwise_batchnorm_beta_read_readvariableopL
Hsavev2_mobilenetv1_conv2d_2_depthwise_batchnorm_beta_read_readvariableopS
Osavev2_mobilenetv1_conv2d_2_pointwise_batchnorm_moving_mean_read_readvariableopL
Hsavev2_mobilenetv1_conv2d_3_pointwise_batchnorm_beta_read_readvariableopS
Osavev2_mobilenetv1_conv2d_6_depthwise_batchnorm_moving_mean_read_readvariableopN
Jsavev2_mobilenetv1_conv2d_10_pointwise_batchnorm_gamma_read_readvariableopN
Jsavev2_mobilenetv1_conv2d_12_depthwise_batchnorm_gamma_read_readvariableopT
Psavev2_mobilenetv1_conv2d_13_depthwise_batchnorm_moving_mean_read_readvariableopO
Ksavev2_mobilenetv1_conv2d_5_depthwise_depthwise_weights_read_readvariableopM
Isavev2_mobilenetv1_conv2d_13_depthwise_batchnorm_beta_read_readvariableopM
Isavev2_mobilenetv1_conv2d_8_pointwise_batchnorm_gamma_read_readvariableopL
Hsavev2_mobilenetv1_conv2d_3_depthwise_batchnorm_beta_read_readvariableopM
Isavev2_mobilenetv1_conv2d_7_depthwise_batchnorm_gamma_read_readvariableopW
Ssavev2_mobilenetv1_conv2d_8_depthwise_batchnorm_moving_variance_read_readvariableopT
Psavev2_mobilenetv1_conv2d_10_depthwise_batchnorm_moving_mean_read_readvariableopX
Tsavev2_mobilenetv1_conv2d_10_pointwise_batchnorm_moving_variance_read_readvariableopF
Bsavev2_mobilenetv1_conv2d_12_pointwise_weights_read_readvariableopX
Tsavev2_mobilenetv1_conv2d_12_pointwise_batchnorm_moving_variance_read_readvariableopM
Isavev2_mobilenetv1_conv2d_4_depthwise_batchnorm_gamma_read_readvariableopS
Osavev2_mobilenetv1_conv2d_8_depthwise_batchnorm_moving_mean_read_readvariableopT
Psavev2_mobilenetv1_conv2d_13_pointwise_batchnorm_moving_mean_read_readvariableopW
Ssavev2_mobilenetv1_conv2d_7_depthwise_batchnorm_moving_variance_read_readvariableopE
Asavev2_mobilenetv1_conv2d_7_pointwise_weights_read_readvariableopE
Asavev2_mobilenetv1_conv2d_5_pointwise_weights_read_readvariableopW
Ssavev2_mobilenetv1_conv2d_1_pointwise_batchnorm_moving_variance_read_readvariableopS
Osavev2_mobilenetv1_conv2d_1_depthwise_batchnorm_moving_mean_read_readvariableopW
Ssavev2_mobilenetv1_conv2d_3_pointwise_batchnorm_moving_variance_read_readvariableopS
Osavev2_mobilenetv1_conv2d_9_pointwise_batchnorm_moving_mean_read_readvariableopM
Isavev2_mobilenetv1_conv2d_10_pointwise_batchnorm_beta_read_readvariableopX
Tsavev2_mobilenetv1_conv2d_12_depthwise_batchnorm_moving_variance_read_readvariableopS
Osavev2_mobilenetv1_conv2d_4_pointwise_batchnorm_moving_mean_read_readvariableopF
Bsavev2_mobilenetv1_conv2d_10_pointwise_weights_read_readvariableopO
Ksavev2_mobilenetv1_conv2d_1_depthwise_depthwise_weights_read_readvariableopL
Hsavev2_mobilenetv1_conv2d_1_pointwise_batchnorm_beta_read_readvariableopM
Isavev2_mobilenetv1_conv2d_0_batchnorm_moving_variance_read_readvariableopM
Isavev2_mobilenetv1_conv2d_2_pointwise_batchnorm_gamma_read_readvariableopO
Ksavev2_mobilenetv1_conv2d_9_depthwise_depthwise_weights_read_readvariableopT
Psavev2_mobilenetv1_conv2d_11_pointwise_batchnorm_moving_mean_read_readvariableopO
Ksavev2_mobilenetv1_conv2d_8_depthwise_depthwise_weights_read_readvariableopL
Hsavev2_mobilenetv1_conv2d_8_depthwise_batchnorm_beta_read_readvariableopW
Ssavev2_mobilenetv1_conv2d_8_pointwise_batchnorm_moving_variance_read_readvariableopE
Asavev2_mobilenetv1_conv2d_3_pointwise_weights_read_readvariableopM
Isavev2_mobilenetv1_conv2d_12_depthwise_batchnorm_beta_read_readvariableopX
Tsavev2_mobilenetv1_conv2d_13_depthwise_batchnorm_moving_variance_read_readvariableopF
Bsavev2_mobilenetv1_conv2d_13_pointwise_weights_read_readvariableopO
Ksavev2_mobilenetv1_conv2d_7_depthwise_depthwise_weights_read_readvariableopL
Hsavev2_mobilenetv1_conv2d_7_depthwise_batchnorm_beta_read_readvariableopN
Jsavev2_mobilenetv1_conv2d_13_pointwise_batchnorm_gamma_read_readvariableopC
?savev2_mobilenetv1_conv2d_0_batchnorm_gamma_read_readvariableopL
Hsavev2_mobilenetv1_conv2d_4_depthwise_batchnorm_beta_read_readvariableopL
Hsavev2_mobilenetv1_conv2d_9_pointwise_batchnorm_beta_read_readvariableopM
Isavev2_mobilenetv1_conv2d_5_pointwise_batchnorm_gamma_read_readvariableopM
Isavev2_mobilenetv1_conv2d_2_depthwise_batchnorm_gamma_read_readvariableopM
Isavev2_mobilenetv1_conv2d_4_pointwise_batchnorm_gamma_read_readvariableopO
Ksavev2_mobilenetv1_conv2d_6_depthwise_depthwise_weights_read_readvariableopM
Isavev2_mobilenetv1_conv2d_9_depthwise_batchnorm_gamma_read_readvariableopX
Tsavev2_mobilenetv1_conv2d_11_depthwise_batchnorm_moving_variance_read_readvariableopN
Jsavev2_mobilenetv1_conv2d_13_depthwise_batchnorm_gamma_read_readvariableopX
Tsavev2_mobilenetv1_conv2d_13_pointwise_batchnorm_moving_variance_read_readvariableopT
Psavev2_mobilenetv1_conv2d_11_depthwise_batchnorm_moving_mean_read_readvariableopE
Asavev2_mobilenetv1_conv2d_8_pointwise_weights_read_readvariableopS
Osavev2_mobilenetv1_conv2d_3_depthwise_batchnorm_moving_mean_read_readvariableopE
Asavev2_mobilenetv1_conv2d_4_pointwise_weights_read_readvariableopW
Ssavev2_mobilenetv1_conv2d_4_pointwise_batchnorm_moving_variance_read_readvariableopL
Hsavev2_mobilenetv1_conv2d_6_depthwise_batchnorm_beta_read_readvariableopW
Ssavev2_mobilenetv1_conv2d_6_pointwise_batchnorm_moving_variance_read_readvariableopS
Osavev2_mobilenetv1_conv2d_9_depthwise_batchnorm_moving_mean_read_readvariableopM
Isavev2_mobilenetv1_conv2d_9_pointwise_batchnorm_gamma_read_readvariableopT
Psavev2_mobilenetv1_conv2d_12_pointwise_batchnorm_moving_mean_read_readvariableopM
Isavev2_mobilenetv1_conv2d_13_pointwise_batchnorm_beta_read_readvariableopS
Osavev2_mobilenetv1_conv2d_7_pointwise_batchnorm_moving_mean_read_readvariableopT
Psavev2_mobilenetv1_conv2d_12_depthwise_batchnorm_moving_mean_read_readvariableopO
Ksavev2_mobilenetv1_conv2d_3_depthwise_depthwise_weights_read_readvariableopL
Hsavev2_mobilenetv1_conv2d_4_pointwise_batchnorm_beta_read_readvariableopS
Osavev2_mobilenetv1_conv2d_2_depthwise_batchnorm_moving_mean_read_readvariableopL
Hsavev2_mobilenetv1_conv2d_2_pointwise_batchnorm_beta_read_readvariableopN
Jsavev2_mobilenetv1_conv2d_10_depthwise_batchnorm_gamma_read_readvariableopP
Lsavev2_mobilenetv1_conv2d_12_depthwise_depthwise_weights_read_readvariableopB
>savev2_mobilenetv1_conv2d_0_batchnorm_beta_read_readvariableopP
Lsavev2_mobilenetv1_conv2d_11_depthwise_depthwise_weights_read_readvariableopM
Isavev2_mobilenetv1_conv2d_11_depthwise_batchnorm_beta_read_readvariableopX
Tsavev2_mobilenetv1_conv2d_11_pointwise_batchnorm_moving_variance_read_readvariableopS
Osavev2_mobilenetv1_conv2d_7_depthwise_batchnorm_moving_mean_read_readvariableopM
Isavev2_mobilenetv1_conv2d_1_depthwise_batchnorm_gamma_read_readvariableopW
Ssavev2_mobilenetv1_conv2d_3_depthwise_batchnorm_moving_variance_read_readvariableopS
Osavev2_mobilenetv1_conv2d_5_pointwise_batchnorm_moving_mean_read_readvariableopN
Jsavev2_mobilenetv1_conv2d_12_pointwise_batchnorm_gamma_read_readvariableopE
Asavev2_mobilenetv1_conv2d_2_pointwise_weights_read_readvariableopW
Ssavev2_mobilenetv1_conv2d_2_pointwise_batchnorm_moving_variance_read_readvariableopM
Isavev2_mobilenetv1_conv2d_5_depthwise_batchnorm_gamma_read_readvariableopP
Lsavev2_mobilenetv1_conv2d_10_depthwise_depthwise_weights_read_readvariableopN
Jsavev2_mobilenetv1_conv2d_11_pointwise_batchnorm_gamma_read_readvariableopM
Isavev2_mobilenetv1_conv2d_6_pointwise_batchnorm_gamma_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?-
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes	
:?*
dtype0*?,
value?,B?,?B'save_counter/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB'variables/32/.ATTRIBUTES/VARIABLE_VALUEB'variables/33/.ATTRIBUTES/VARIABLE_VALUEB'variables/34/.ATTRIBUTES/VARIABLE_VALUEB'variables/35/.ATTRIBUTES/VARIABLE_VALUEB'variables/36/.ATTRIBUTES/VARIABLE_VALUEB'variables/37/.ATTRIBUTES/VARIABLE_VALUEB'variables/38/.ATTRIBUTES/VARIABLE_VALUEB'variables/39/.ATTRIBUTES/VARIABLE_VALUEB'variables/40/.ATTRIBUTES/VARIABLE_VALUEB'variables/41/.ATTRIBUTES/VARIABLE_VALUEB'variables/42/.ATTRIBUTES/VARIABLE_VALUEB'variables/43/.ATTRIBUTES/VARIABLE_VALUEB'variables/44/.ATTRIBUTES/VARIABLE_VALUEB'variables/45/.ATTRIBUTES/VARIABLE_VALUEB'variables/46/.ATTRIBUTES/VARIABLE_VALUEB'variables/47/.ATTRIBUTES/VARIABLE_VALUEB'variables/48/.ATTRIBUTES/VARIABLE_VALUEB'variables/49/.ATTRIBUTES/VARIABLE_VALUEB'variables/50/.ATTRIBUTES/VARIABLE_VALUEB'variables/51/.ATTRIBUTES/VARIABLE_VALUEB'variables/52/.ATTRIBUTES/VARIABLE_VALUEB'variables/53/.ATTRIBUTES/VARIABLE_VALUEB'variables/54/.ATTRIBUTES/VARIABLE_VALUEB'variables/55/.ATTRIBUTES/VARIABLE_VALUEB'variables/56/.ATTRIBUTES/VARIABLE_VALUEB'variables/57/.ATTRIBUTES/VARIABLE_VALUEB'variables/58/.ATTRIBUTES/VARIABLE_VALUEB'variables/59/.ATTRIBUTES/VARIABLE_VALUEB'variables/60/.ATTRIBUTES/VARIABLE_VALUEB'variables/61/.ATTRIBUTES/VARIABLE_VALUEB'variables/62/.ATTRIBUTES/VARIABLE_VALUEB'variables/63/.ATTRIBUTES/VARIABLE_VALUEB'variables/64/.ATTRIBUTES/VARIABLE_VALUEB'variables/65/.ATTRIBUTES/VARIABLE_VALUEB'variables/66/.ATTRIBUTES/VARIABLE_VALUEB'variables/67/.ATTRIBUTES/VARIABLE_VALUEB'variables/68/.ATTRIBUTES/VARIABLE_VALUEB'variables/69/.ATTRIBUTES/VARIABLE_VALUEB'variables/70/.ATTRIBUTES/VARIABLE_VALUEB'variables/71/.ATTRIBUTES/VARIABLE_VALUEB'variables/72/.ATTRIBUTES/VARIABLE_VALUEB'variables/73/.ATTRIBUTES/VARIABLE_VALUEB'variables/74/.ATTRIBUTES/VARIABLE_VALUEB'variables/75/.ATTRIBUTES/VARIABLE_VALUEB'variables/76/.ATTRIBUTES/VARIABLE_VALUEB'variables/77/.ATTRIBUTES/VARIABLE_VALUEB'variables/78/.ATTRIBUTES/VARIABLE_VALUEB'variables/79/.ATTRIBUTES/VARIABLE_VALUEB'variables/80/.ATTRIBUTES/VARIABLE_VALUEB'variables/81/.ATTRIBUTES/VARIABLE_VALUEB'variables/82/.ATTRIBUTES/VARIABLE_VALUEB'variables/83/.ATTRIBUTES/VARIABLE_VALUEB'variables/84/.ATTRIBUTES/VARIABLE_VALUEB'variables/85/.ATTRIBUTES/VARIABLE_VALUEB'variables/86/.ATTRIBUTES/VARIABLE_VALUEB'variables/87/.ATTRIBUTES/VARIABLE_VALUEB'variables/88/.ATTRIBUTES/VARIABLE_VALUEB'variables/89/.ATTRIBUTES/VARIABLE_VALUEB'variables/90/.ATTRIBUTES/VARIABLE_VALUEB'variables/91/.ATTRIBUTES/VARIABLE_VALUEB'variables/92/.ATTRIBUTES/VARIABLE_VALUEB'variables/93/.ATTRIBUTES/VARIABLE_VALUEB'variables/94/.ATTRIBUTES/VARIABLE_VALUEB'variables/95/.ATTRIBUTES/VARIABLE_VALUEB'variables/96/.ATTRIBUTES/VARIABLE_VALUEB'variables/97/.ATTRIBUTES/VARIABLE_VALUEB'variables/98/.ATTRIBUTES/VARIABLE_VALUEB'variables/99/.ATTRIBUTES/VARIABLE_VALUEB(variables/100/.ATTRIBUTES/VARIABLE_VALUEB(variables/101/.ATTRIBUTES/VARIABLE_VALUEB(variables/102/.ATTRIBUTES/VARIABLE_VALUEB(variables/103/.ATTRIBUTES/VARIABLE_VALUEB(variables/104/.ATTRIBUTES/VARIABLE_VALUEB(variables/105/.ATTRIBUTES/VARIABLE_VALUEB(variables/106/.ATTRIBUTES/VARIABLE_VALUEB(variables/107/.ATTRIBUTES/VARIABLE_VALUEB(variables/108/.ATTRIBUTES/VARIABLE_VALUEB(variables/109/.ATTRIBUTES/VARIABLE_VALUEB(variables/110/.ATTRIBUTES/VARIABLE_VALUEB(variables/111/.ATTRIBUTES/VARIABLE_VALUEB(variables/112/.ATTRIBUTES/VARIABLE_VALUEB(variables/113/.ATTRIBUTES/VARIABLE_VALUEB(variables/114/.ATTRIBUTES/VARIABLE_VALUEB(variables/115/.ATTRIBUTES/VARIABLE_VALUEB(variables/116/.ATTRIBUTES/VARIABLE_VALUEB(variables/117/.ATTRIBUTES/VARIABLE_VALUEB(variables/118/.ATTRIBUTES/VARIABLE_VALUEB(variables/119/.ATTRIBUTES/VARIABLE_VALUEB(variables/120/.ATTRIBUTES/VARIABLE_VALUEB(variables/121/.ATTRIBUTES/VARIABLE_VALUEB(variables/122/.ATTRIBUTES/VARIABLE_VALUEB(variables/123/.ATTRIBUTES/VARIABLE_VALUEB(variables/124/.ATTRIBUTES/VARIABLE_VALUEB(variables/125/.ATTRIBUTES/VARIABLE_VALUEB(variables/126/.ATTRIBUTES/VARIABLE_VALUEB(variables/127/.ATTRIBUTES/VARIABLE_VALUEB(variables/128/.ATTRIBUTES/VARIABLE_VALUEB(variables/129/.ATTRIBUTES/VARIABLE_VALUEB(variables/130/.ATTRIBUTES/VARIABLE_VALUEB(variables/131/.ATTRIBUTES/VARIABLE_VALUEB(variables/132/.ATTRIBUTES/VARIABLE_VALUEB(variables/133/.ATTRIBUTES/VARIABLE_VALUEB(variables/134/.ATTRIBUTES/VARIABLE_VALUEB(variables/135/.ATTRIBUTES/VARIABLE_VALUEB(variables/136/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes	
:?*
dtype0*?
value?B??B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?U
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_save_counter_read_readvariableopSsavev2_mobilenetv1_conv2d_1_depthwise_batchnorm_moving_variance_read_readvariableopIsavev2_mobilenetv1_conv2d_3_depthwise_batchnorm_gamma_read_readvariableopIsavev2_mobilenetv1_conv2d_7_pointwise_batchnorm_gamma_read_readvariableopHsavev2_mobilenetv1_conv2d_8_pointwise_batchnorm_beta_read_readvariableopIsavev2_mobilenetv1_conv2d_11_pointwise_batchnorm_beta_read_readvariableopKsavev2_mobilenetv1_conv2d_4_depthwise_depthwise_weights_read_readvariableopCsavev2_mobilenetv1_logits_conv2d_1c_1x1_weights_read_readvariableopEsavev2_mobilenetv1_conv2d_0_batchnorm_moving_mean_read_readvariableopOsavev2_mobilenetv1_conv2d_3_pointwise_batchnorm_moving_mean_read_readvariableopOsavev2_mobilenetv1_conv2d_4_depthwise_batchnorm_moving_mean_read_readvariableopSsavev2_mobilenetv1_conv2d_5_depthwise_batchnorm_moving_variance_read_readvariableopIsavev2_mobilenetv1_conv2d_6_depthwise_batchnorm_gamma_read_readvariableopSsavev2_mobilenetv1_conv2d_7_pointwise_batchnorm_moving_variance_read_readvariableopSsavev2_mobilenetv1_conv2d_9_depthwise_batchnorm_moving_variance_read_readvariableopSsavev2_mobilenetv1_conv2d_9_pointwise_batchnorm_moving_variance_read_readvariableopBsavev2_mobilenetv1_conv2d_11_pointwise_weights_read_readvariableopOsavev2_mobilenetv1_conv2d_1_pointwise_batchnorm_moving_mean_read_readvariableopOsavev2_mobilenetv1_conv2d_5_depthwise_batchnorm_moving_mean_read_readvariableopPsavev2_mobilenetv1_conv2d_10_pointwise_batchnorm_moving_mean_read_readvariableopHsavev2_mobilenetv1_conv2d_7_pointwise_batchnorm_beta_read_readvariableop7savev2_mobilenetv1_conv2d_0_weights_read_readvariableopSsavev2_mobilenetv1_conv2d_2_depthwise_batchnorm_moving_variance_read_readvariableopSsavev2_mobilenetv1_conv2d_4_depthwise_batchnorm_moving_variance_read_readvariableopOsavev2_mobilenetv1_conv2d_6_pointwise_batchnorm_moving_mean_read_readvariableopHsavev2_mobilenetv1_conv2d_9_depthwise_batchnorm_beta_read_readvariableopJsavev2_mobilenetv1_conv2d_11_depthwise_batchnorm_gamma_read_readvariableopAsavev2_mobilenetv1_conv2d_6_pointwise_weights_read_readvariableopHsavev2_mobilenetv1_conv2d_6_pointwise_batchnorm_beta_read_readvariableopHsavev2_mobilenetv1_conv2d_1_depthwise_batchnorm_beta_read_readvariableopIsavev2_mobilenetv1_conv2d_3_pointwise_batchnorm_gamma_read_readvariableopIsavev2_mobilenetv1_conv2d_10_depthwise_batchnorm_beta_read_readvariableopBsavev2_mobilenetv1_logits_conv2d_1c_1x1_biases_read_readvariableopOsavev2_mobilenetv1_conv2d_8_pointwise_batchnorm_moving_mean_read_readvariableopHsavev2_mobilenetv1_conv2d_5_pointwise_batchnorm_beta_read_readvariableopKsavev2_mobilenetv1_conv2d_2_depthwise_depthwise_weights_read_readvariableopAsavev2_mobilenetv1_conv2d_1_pointwise_weights_read_readvariableopIsavev2_mobilenetv1_conv2d_1_pointwise_batchnorm_gamma_read_readvariableopHsavev2_mobilenetv1_conv2d_5_depthwise_batchnorm_beta_read_readvariableopSsavev2_mobilenetv1_conv2d_5_pointwise_batchnorm_moving_variance_read_readvariableopIsavev2_mobilenetv1_conv2d_8_depthwise_batchnorm_gamma_read_readvariableopLsavev2_mobilenetv1_conv2d_13_depthwise_depthwise_weights_read_readvariableopSsavev2_mobilenetv1_conv2d_6_depthwise_batchnorm_moving_variance_read_readvariableopAsavev2_mobilenetv1_conv2d_9_pointwise_weights_read_readvariableopTsavev2_mobilenetv1_conv2d_10_depthwise_batchnorm_moving_variance_read_readvariableopIsavev2_mobilenetv1_conv2d_12_pointwise_batchnorm_beta_read_readvariableopHsavev2_mobilenetv1_conv2d_2_depthwise_batchnorm_beta_read_readvariableopOsavev2_mobilenetv1_conv2d_2_pointwise_batchnorm_moving_mean_read_readvariableopHsavev2_mobilenetv1_conv2d_3_pointwise_batchnorm_beta_read_readvariableopOsavev2_mobilenetv1_conv2d_6_depthwise_batchnorm_moving_mean_read_readvariableopJsavev2_mobilenetv1_conv2d_10_pointwise_batchnorm_gamma_read_readvariableopJsavev2_mobilenetv1_conv2d_12_depthwise_batchnorm_gamma_read_readvariableopPsavev2_mobilenetv1_conv2d_13_depthwise_batchnorm_moving_mean_read_readvariableopKsavev2_mobilenetv1_conv2d_5_depthwise_depthwise_weights_read_readvariableopIsavev2_mobilenetv1_conv2d_13_depthwise_batchnorm_beta_read_readvariableopIsavev2_mobilenetv1_conv2d_8_pointwise_batchnorm_gamma_read_readvariableopHsavev2_mobilenetv1_conv2d_3_depthwise_batchnorm_beta_read_readvariableopIsavev2_mobilenetv1_conv2d_7_depthwise_batchnorm_gamma_read_readvariableopSsavev2_mobilenetv1_conv2d_8_depthwise_batchnorm_moving_variance_read_readvariableopPsavev2_mobilenetv1_conv2d_10_depthwise_batchnorm_moving_mean_read_readvariableopTsavev2_mobilenetv1_conv2d_10_pointwise_batchnorm_moving_variance_read_readvariableopBsavev2_mobilenetv1_conv2d_12_pointwise_weights_read_readvariableopTsavev2_mobilenetv1_conv2d_12_pointwise_batchnorm_moving_variance_read_readvariableopIsavev2_mobilenetv1_conv2d_4_depthwise_batchnorm_gamma_read_readvariableopOsavev2_mobilenetv1_conv2d_8_depthwise_batchnorm_moving_mean_read_readvariableopPsavev2_mobilenetv1_conv2d_13_pointwise_batchnorm_moving_mean_read_readvariableopSsavev2_mobilenetv1_conv2d_7_depthwise_batchnorm_moving_variance_read_readvariableopAsavev2_mobilenetv1_conv2d_7_pointwise_weights_read_readvariableopAsavev2_mobilenetv1_conv2d_5_pointwise_weights_read_readvariableopSsavev2_mobilenetv1_conv2d_1_pointwise_batchnorm_moving_variance_read_readvariableopOsavev2_mobilenetv1_conv2d_1_depthwise_batchnorm_moving_mean_read_readvariableopSsavev2_mobilenetv1_conv2d_3_pointwise_batchnorm_moving_variance_read_readvariableopOsavev2_mobilenetv1_conv2d_9_pointwise_batchnorm_moving_mean_read_readvariableopIsavev2_mobilenetv1_conv2d_10_pointwise_batchnorm_beta_read_readvariableopTsavev2_mobilenetv1_conv2d_12_depthwise_batchnorm_moving_variance_read_readvariableopOsavev2_mobilenetv1_conv2d_4_pointwise_batchnorm_moving_mean_read_readvariableopBsavev2_mobilenetv1_conv2d_10_pointwise_weights_read_readvariableopKsavev2_mobilenetv1_conv2d_1_depthwise_depthwise_weights_read_readvariableopHsavev2_mobilenetv1_conv2d_1_pointwise_batchnorm_beta_read_readvariableopIsavev2_mobilenetv1_conv2d_0_batchnorm_moving_variance_read_readvariableopIsavev2_mobilenetv1_conv2d_2_pointwise_batchnorm_gamma_read_readvariableopKsavev2_mobilenetv1_conv2d_9_depthwise_depthwise_weights_read_readvariableopPsavev2_mobilenetv1_conv2d_11_pointwise_batchnorm_moving_mean_read_readvariableopKsavev2_mobilenetv1_conv2d_8_depthwise_depthwise_weights_read_readvariableopHsavev2_mobilenetv1_conv2d_8_depthwise_batchnorm_beta_read_readvariableopSsavev2_mobilenetv1_conv2d_8_pointwise_batchnorm_moving_variance_read_readvariableopAsavev2_mobilenetv1_conv2d_3_pointwise_weights_read_readvariableopIsavev2_mobilenetv1_conv2d_12_depthwise_batchnorm_beta_read_readvariableopTsavev2_mobilenetv1_conv2d_13_depthwise_batchnorm_moving_variance_read_readvariableopBsavev2_mobilenetv1_conv2d_13_pointwise_weights_read_readvariableopKsavev2_mobilenetv1_conv2d_7_depthwise_depthwise_weights_read_readvariableopHsavev2_mobilenetv1_conv2d_7_depthwise_batchnorm_beta_read_readvariableopJsavev2_mobilenetv1_conv2d_13_pointwise_batchnorm_gamma_read_readvariableop?savev2_mobilenetv1_conv2d_0_batchnorm_gamma_read_readvariableopHsavev2_mobilenetv1_conv2d_4_depthwise_batchnorm_beta_read_readvariableopHsavev2_mobilenetv1_conv2d_9_pointwise_batchnorm_beta_read_readvariableopIsavev2_mobilenetv1_conv2d_5_pointwise_batchnorm_gamma_read_readvariableopIsavev2_mobilenetv1_conv2d_2_depthwise_batchnorm_gamma_read_readvariableopIsavev2_mobilenetv1_conv2d_4_pointwise_batchnorm_gamma_read_readvariableopKsavev2_mobilenetv1_conv2d_6_depthwise_depthwise_weights_read_readvariableopIsavev2_mobilenetv1_conv2d_9_depthwise_batchnorm_gamma_read_readvariableopTsavev2_mobilenetv1_conv2d_11_depthwise_batchnorm_moving_variance_read_readvariableopJsavev2_mobilenetv1_conv2d_13_depthwise_batchnorm_gamma_read_readvariableopTsavev2_mobilenetv1_conv2d_13_pointwise_batchnorm_moving_variance_read_readvariableopPsavev2_mobilenetv1_conv2d_11_depthwise_batchnorm_moving_mean_read_readvariableopAsavev2_mobilenetv1_conv2d_8_pointwise_weights_read_readvariableopOsavev2_mobilenetv1_conv2d_3_depthwise_batchnorm_moving_mean_read_readvariableopAsavev2_mobilenetv1_conv2d_4_pointwise_weights_read_readvariableopSsavev2_mobilenetv1_conv2d_4_pointwise_batchnorm_moving_variance_read_readvariableopHsavev2_mobilenetv1_conv2d_6_depthwise_batchnorm_beta_read_readvariableopSsavev2_mobilenetv1_conv2d_6_pointwise_batchnorm_moving_variance_read_readvariableopOsavev2_mobilenetv1_conv2d_9_depthwise_batchnorm_moving_mean_read_readvariableopIsavev2_mobilenetv1_conv2d_9_pointwise_batchnorm_gamma_read_readvariableopPsavev2_mobilenetv1_conv2d_12_pointwise_batchnorm_moving_mean_read_readvariableopIsavev2_mobilenetv1_conv2d_13_pointwise_batchnorm_beta_read_readvariableopOsavev2_mobilenetv1_conv2d_7_pointwise_batchnorm_moving_mean_read_readvariableopPsavev2_mobilenetv1_conv2d_12_depthwise_batchnorm_moving_mean_read_readvariableopKsavev2_mobilenetv1_conv2d_3_depthwise_depthwise_weights_read_readvariableopHsavev2_mobilenetv1_conv2d_4_pointwise_batchnorm_beta_read_readvariableopOsavev2_mobilenetv1_conv2d_2_depthwise_batchnorm_moving_mean_read_readvariableopHsavev2_mobilenetv1_conv2d_2_pointwise_batchnorm_beta_read_readvariableopJsavev2_mobilenetv1_conv2d_10_depthwise_batchnorm_gamma_read_readvariableopLsavev2_mobilenetv1_conv2d_12_depthwise_depthwise_weights_read_readvariableop>savev2_mobilenetv1_conv2d_0_batchnorm_beta_read_readvariableopLsavev2_mobilenetv1_conv2d_11_depthwise_depthwise_weights_read_readvariableopIsavev2_mobilenetv1_conv2d_11_depthwise_batchnorm_beta_read_readvariableopTsavev2_mobilenetv1_conv2d_11_pointwise_batchnorm_moving_variance_read_readvariableopOsavev2_mobilenetv1_conv2d_7_depthwise_batchnorm_moving_mean_read_readvariableopIsavev2_mobilenetv1_conv2d_1_depthwise_batchnorm_gamma_read_readvariableopSsavev2_mobilenetv1_conv2d_3_depthwise_batchnorm_moving_variance_read_readvariableopOsavev2_mobilenetv1_conv2d_5_pointwise_batchnorm_moving_mean_read_readvariableopJsavev2_mobilenetv1_conv2d_12_pointwise_batchnorm_gamma_read_readvariableopAsavev2_mobilenetv1_conv2d_2_pointwise_weights_read_readvariableopSsavev2_mobilenetv1_conv2d_2_pointwise_batchnorm_moving_variance_read_readvariableopIsavev2_mobilenetv1_conv2d_5_depthwise_batchnorm_gamma_read_readvariableopLsavev2_mobilenetv1_conv2d_10_depthwise_depthwise_weights_read_readvariableopJsavev2_mobilenetv1_conv2d_11_pointwise_batchnorm_gamma_read_readvariableopIsavev2_mobilenetv1_conv2d_6_pointwise_batchnorm_gamma_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *?
dtypes?
?2?	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1?
NoOpNoOp^MergeV2Checkpoints*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*?

_input_shapes?	
?	: : ::`:?:?:?:`:??::`:`:?:?:?:?:?:??:0:?:?:?::0:`:?:?:?:??:?::`:?:?:?:?:0:0:0:?:?:?:?:?:??:?:?:0:`:`:?:?:?:?:?:?:?:`:?:?:?:?:??:?:`:?:?:?:??:??:0::`:?:?:?:?:??::0::`:?:?:?:?:?:``:?:?:??:?:?:?::`:?:?:0:?:?:?:?:?:?:?:??:`:`?:?:?:?:?:?:?:?:?:?:`:?:0:`:?:?::?:?:?:?::`:?:?:0`:`:?:?:?:?: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:`:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:,(
&
_output_shapes
:`:.*
(
_output_shapes
:??: 	

_output_shapes
:: 


_output_shapes
:`: 

_output_shapes
:`:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:.*
(
_output_shapes
:??: 

_output_shapes
:0:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:,(
&
_output_shapes
:: 

_output_shapes
:0: 

_output_shapes
:`:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?: 

_output_shapes
:: 

_output_shapes
:`:! 

_output_shapes	
:?:!!

_output_shapes	
:?:!"

_output_shapes	
:?:!#

_output_shapes	
:?:,$(
&
_output_shapes
:0:,%(
&
_output_shapes
:0: &

_output_shapes
:0:!'

_output_shapes	
:?:!(

_output_shapes	
:?:!)

_output_shapes	
:?:-*)
'
_output_shapes
:?:!+

_output_shapes	
:?:.,*
(
_output_shapes
:??:!-

_output_shapes	
:?:!.

_output_shapes	
:?: /

_output_shapes
:0: 0

_output_shapes
:`: 1

_output_shapes
:`:!2

_output_shapes	
:?:!3

_output_shapes	
:?:!4

_output_shapes	
:?:!5

_output_shapes	
:?:-6)
'
_output_shapes
:?:!7

_output_shapes	
:?:!8

_output_shapes	
:?: 9

_output_shapes
:`:!:

_output_shapes	
:?:!;

_output_shapes	
:?:!<

_output_shapes	
:?:!=

_output_shapes	
:?:.>*
(
_output_shapes
:??:!?

_output_shapes	
:?: @

_output_shapes
:`:!A

_output_shapes	
:?:!B

_output_shapes	
:?:!C

_output_shapes	
:?:.D*
(
_output_shapes
:??:.E*
(
_output_shapes
:??: F

_output_shapes
:0: G

_output_shapes
:: H

_output_shapes
:`:!I

_output_shapes	
:?:!J

_output_shapes	
:?:!K

_output_shapes	
:?:!L

_output_shapes	
:?:.M*
(
_output_shapes
:??:,N(
&
_output_shapes
:: O

_output_shapes
:0: P

_output_shapes
:: Q

_output_shapes
:`:-R)
'
_output_shapes
:?:!S

_output_shapes	
:?:-T)
'
_output_shapes
:?:!U

_output_shapes	
:?:!V

_output_shapes	
:?:,W(
&
_output_shapes
:``:!X

_output_shapes	
:?:!Y

_output_shapes	
:?:.Z*
(
_output_shapes
:??:-[)
'
_output_shapes
:?:!\

_output_shapes	
:?:!]

_output_shapes	
:?: ^

_output_shapes
:: _

_output_shapes
:`:!`

_output_shapes	
:?:!a

_output_shapes	
:?: b

_output_shapes
:0:!c

_output_shapes	
:?:-d)
'
_output_shapes
:?:!e

_output_shapes	
:?:!f

_output_shapes	
:?:!g

_output_shapes	
:?:!h

_output_shapes	
:?:!i

_output_shapes	
:?:.j*
(
_output_shapes
:??: k

_output_shapes
:`:-l)
'
_output_shapes
:`?:!m

_output_shapes	
:?:!n

_output_shapes	
:?:!o

_output_shapes	
:?:!p

_output_shapes	
:?:!q

_output_shapes	
:?:!r

_output_shapes	
:?:!s

_output_shapes	
:?:!t

_output_shapes	
:?:!u

_output_shapes	
:?:,v(
&
_output_shapes
:`:!w

_output_shapes	
:?: x

_output_shapes
:0: y

_output_shapes
:`:!z

_output_shapes	
:?:-{)
'
_output_shapes
:?: |

_output_shapes
::-})
'
_output_shapes
:?:!~

_output_shapes	
:?:!

_output_shapes	
:?:"?

_output_shapes	
:?:!?

_output_shapes
::!?

_output_shapes
:`:"?

_output_shapes	
:?:"?

_output_shapes	
:?:-?(
&
_output_shapes
:0`:!?

_output_shapes
:`:"?

_output_shapes	
:?:.?)
'
_output_shapes
:?:"?

_output_shapes	
:?:"?

_output_shapes	
:?:?

_output_shapes
: 
??
?x
__inference_pruned_3593?
gtrain_mobilenetv1_mobilenetv1_conv2d_0_kernel_regularizer_l2_regularizer_l2loss_readvariableop_resource:V
Htrain_mobilenetv1_mobilenetv1_conv2d_0_batchnorm_readvariableop_resource:X
Jtrain_mobilenetv1_mobilenetv1_conv2d_0_batchnorm_readvariableop_1_resource:g
Ytrain_mobilenetv1_mobilenetv1_conv2d_0_batchnorm_fusedbatchnormv3_readvariableop_resource:i
[train_mobilenetv1_mobilenetv1_conv2d_0_batchnorm_fusedbatchnormv3_readvariableop_1_resource:l
Rtrain_mobilenetv1_mobilenetv1_conv2d_1_depthwise_depthwise_readvariableop_resource:`
Rtrain_mobilenetv1_mobilenetv1_conv2d_1_depthwise_batchnorm_readvariableop_resource:b
Ttrain_mobilenetv1_mobilenetv1_conv2d_1_depthwise_batchnorm_readvariableop_1_resource:q
ctrain_mobilenetv1_mobilenetv1_conv2d_1_depthwise_batchnorm_fusedbatchnormv3_readvariableop_resource:s
etrain_mobilenetv1_mobilenetv1_conv2d_1_depthwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource:?
qtrain_mobilenetv1_mobilenetv1_conv2d_1_pointwise_kernel_regularizer_l2_regularizer_l2loss_readvariableop_resource:0`
Rtrain_mobilenetv1_mobilenetv1_conv2d_1_pointwise_batchnorm_readvariableop_resource:0b
Ttrain_mobilenetv1_mobilenetv1_conv2d_1_pointwise_batchnorm_readvariableop_1_resource:0q
ctrain_mobilenetv1_mobilenetv1_conv2d_1_pointwise_batchnorm_fusedbatchnormv3_readvariableop_resource:0s
etrain_mobilenetv1_mobilenetv1_conv2d_1_pointwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource:0l
Rtrain_mobilenetv1_mobilenetv1_conv2d_2_depthwise_depthwise_readvariableop_resource:0`
Rtrain_mobilenetv1_mobilenetv1_conv2d_2_depthwise_batchnorm_readvariableop_resource:0b
Ttrain_mobilenetv1_mobilenetv1_conv2d_2_depthwise_batchnorm_readvariableop_1_resource:0q
ctrain_mobilenetv1_mobilenetv1_conv2d_2_depthwise_batchnorm_fusedbatchnormv3_readvariableop_resource:0s
etrain_mobilenetv1_mobilenetv1_conv2d_2_depthwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource:0?
qtrain_mobilenetv1_mobilenetv1_conv2d_2_pointwise_kernel_regularizer_l2_regularizer_l2loss_readvariableop_resource:0``
Rtrain_mobilenetv1_mobilenetv1_conv2d_2_pointwise_batchnorm_readvariableop_resource:`b
Ttrain_mobilenetv1_mobilenetv1_conv2d_2_pointwise_batchnorm_readvariableop_1_resource:`q
ctrain_mobilenetv1_mobilenetv1_conv2d_2_pointwise_batchnorm_fusedbatchnormv3_readvariableop_resource:`s
etrain_mobilenetv1_mobilenetv1_conv2d_2_pointwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource:`l
Rtrain_mobilenetv1_mobilenetv1_conv2d_3_depthwise_depthwise_readvariableop_resource:``
Rtrain_mobilenetv1_mobilenetv1_conv2d_3_depthwise_batchnorm_readvariableop_resource:`b
Ttrain_mobilenetv1_mobilenetv1_conv2d_3_depthwise_batchnorm_readvariableop_1_resource:`q
ctrain_mobilenetv1_mobilenetv1_conv2d_3_depthwise_batchnorm_fusedbatchnormv3_readvariableop_resource:`s
etrain_mobilenetv1_mobilenetv1_conv2d_3_depthwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource:`?
qtrain_mobilenetv1_mobilenetv1_conv2d_3_pointwise_kernel_regularizer_l2_regularizer_l2loss_readvariableop_resource:```
Rtrain_mobilenetv1_mobilenetv1_conv2d_3_pointwise_batchnorm_readvariableop_resource:`b
Ttrain_mobilenetv1_mobilenetv1_conv2d_3_pointwise_batchnorm_readvariableop_1_resource:`q
ctrain_mobilenetv1_mobilenetv1_conv2d_3_pointwise_batchnorm_fusedbatchnormv3_readvariableop_resource:`s
etrain_mobilenetv1_mobilenetv1_conv2d_3_pointwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource:`l
Rtrain_mobilenetv1_mobilenetv1_conv2d_4_depthwise_depthwise_readvariableop_resource:``
Rtrain_mobilenetv1_mobilenetv1_conv2d_4_depthwise_batchnorm_readvariableop_resource:`b
Ttrain_mobilenetv1_mobilenetv1_conv2d_4_depthwise_batchnorm_readvariableop_1_resource:`q
ctrain_mobilenetv1_mobilenetv1_conv2d_4_depthwise_batchnorm_fusedbatchnormv3_readvariableop_resource:`s
etrain_mobilenetv1_mobilenetv1_conv2d_4_depthwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource:`?
qtrain_mobilenetv1_mobilenetv1_conv2d_4_pointwise_kernel_regularizer_l2_regularizer_l2loss_readvariableop_resource:`?a
Rtrain_mobilenetv1_mobilenetv1_conv2d_4_pointwise_batchnorm_readvariableop_resource:	?c
Ttrain_mobilenetv1_mobilenetv1_conv2d_4_pointwise_batchnorm_readvariableop_1_resource:	?r
ctrain_mobilenetv1_mobilenetv1_conv2d_4_pointwise_batchnorm_fusedbatchnormv3_readvariableop_resource:	?t
etrain_mobilenetv1_mobilenetv1_conv2d_4_pointwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource:	?m
Rtrain_mobilenetv1_mobilenetv1_conv2d_5_depthwise_depthwise_readvariableop_resource:?a
Rtrain_mobilenetv1_mobilenetv1_conv2d_5_depthwise_batchnorm_readvariableop_resource:	?c
Ttrain_mobilenetv1_mobilenetv1_conv2d_5_depthwise_batchnorm_readvariableop_1_resource:	?r
ctrain_mobilenetv1_mobilenetv1_conv2d_5_depthwise_batchnorm_fusedbatchnormv3_readvariableop_resource:	?t
etrain_mobilenetv1_mobilenetv1_conv2d_5_depthwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource:	??
qtrain_mobilenetv1_mobilenetv1_conv2d_5_pointwise_kernel_regularizer_l2_regularizer_l2loss_readvariableop_resource:??a
Rtrain_mobilenetv1_mobilenetv1_conv2d_5_pointwise_batchnorm_readvariableop_resource:	?c
Ttrain_mobilenetv1_mobilenetv1_conv2d_5_pointwise_batchnorm_readvariableop_1_resource:	?r
ctrain_mobilenetv1_mobilenetv1_conv2d_5_pointwise_batchnorm_fusedbatchnormv3_readvariableop_resource:	?t
etrain_mobilenetv1_mobilenetv1_conv2d_5_pointwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource:	?m
Rtrain_mobilenetv1_mobilenetv1_conv2d_6_depthwise_depthwise_readvariableop_resource:?a
Rtrain_mobilenetv1_mobilenetv1_conv2d_6_depthwise_batchnorm_readvariableop_resource:	?c
Ttrain_mobilenetv1_mobilenetv1_conv2d_6_depthwise_batchnorm_readvariableop_1_resource:	?r
ctrain_mobilenetv1_mobilenetv1_conv2d_6_depthwise_batchnorm_fusedbatchnormv3_readvariableop_resource:	?t
etrain_mobilenetv1_mobilenetv1_conv2d_6_depthwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource:	??
qtrain_mobilenetv1_mobilenetv1_conv2d_6_pointwise_kernel_regularizer_l2_regularizer_l2loss_readvariableop_resource:??a
Rtrain_mobilenetv1_mobilenetv1_conv2d_6_pointwise_batchnorm_readvariableop_resource:	?c
Ttrain_mobilenetv1_mobilenetv1_conv2d_6_pointwise_batchnorm_readvariableop_1_resource:	?r
ctrain_mobilenetv1_mobilenetv1_conv2d_6_pointwise_batchnorm_fusedbatchnormv3_readvariableop_resource:	?t
etrain_mobilenetv1_mobilenetv1_conv2d_6_pointwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource:	?m
Rtrain_mobilenetv1_mobilenetv1_conv2d_7_depthwise_depthwise_readvariableop_resource:?a
Rtrain_mobilenetv1_mobilenetv1_conv2d_7_depthwise_batchnorm_readvariableop_resource:	?c
Ttrain_mobilenetv1_mobilenetv1_conv2d_7_depthwise_batchnorm_readvariableop_1_resource:	?r
ctrain_mobilenetv1_mobilenetv1_conv2d_7_depthwise_batchnorm_fusedbatchnormv3_readvariableop_resource:	?t
etrain_mobilenetv1_mobilenetv1_conv2d_7_depthwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource:	??
qtrain_mobilenetv1_mobilenetv1_conv2d_7_pointwise_kernel_regularizer_l2_regularizer_l2loss_readvariableop_resource:??a
Rtrain_mobilenetv1_mobilenetv1_conv2d_7_pointwise_batchnorm_readvariableop_resource:	?c
Ttrain_mobilenetv1_mobilenetv1_conv2d_7_pointwise_batchnorm_readvariableop_1_resource:	?r
ctrain_mobilenetv1_mobilenetv1_conv2d_7_pointwise_batchnorm_fusedbatchnormv3_readvariableop_resource:	?t
etrain_mobilenetv1_mobilenetv1_conv2d_7_pointwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource:	?m
Rtrain_mobilenetv1_mobilenetv1_conv2d_8_depthwise_depthwise_readvariableop_resource:?a
Rtrain_mobilenetv1_mobilenetv1_conv2d_8_depthwise_batchnorm_readvariableop_resource:	?c
Ttrain_mobilenetv1_mobilenetv1_conv2d_8_depthwise_batchnorm_readvariableop_1_resource:	?r
ctrain_mobilenetv1_mobilenetv1_conv2d_8_depthwise_batchnorm_fusedbatchnormv3_readvariableop_resource:	?t
etrain_mobilenetv1_mobilenetv1_conv2d_8_depthwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource:	??
qtrain_mobilenetv1_mobilenetv1_conv2d_8_pointwise_kernel_regularizer_l2_regularizer_l2loss_readvariableop_resource:??a
Rtrain_mobilenetv1_mobilenetv1_conv2d_8_pointwise_batchnorm_readvariableop_resource:	?c
Ttrain_mobilenetv1_mobilenetv1_conv2d_8_pointwise_batchnorm_readvariableop_1_resource:	?r
ctrain_mobilenetv1_mobilenetv1_conv2d_8_pointwise_batchnorm_fusedbatchnormv3_readvariableop_resource:	?t
etrain_mobilenetv1_mobilenetv1_conv2d_8_pointwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource:	?m
Rtrain_mobilenetv1_mobilenetv1_conv2d_9_depthwise_depthwise_readvariableop_resource:?a
Rtrain_mobilenetv1_mobilenetv1_conv2d_9_depthwise_batchnorm_readvariableop_resource:	?c
Ttrain_mobilenetv1_mobilenetv1_conv2d_9_depthwise_batchnorm_readvariableop_1_resource:	?r
ctrain_mobilenetv1_mobilenetv1_conv2d_9_depthwise_batchnorm_fusedbatchnormv3_readvariableop_resource:	?t
etrain_mobilenetv1_mobilenetv1_conv2d_9_depthwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource:	??
qtrain_mobilenetv1_mobilenetv1_conv2d_9_pointwise_kernel_regularizer_l2_regularizer_l2loss_readvariableop_resource:??a
Rtrain_mobilenetv1_mobilenetv1_conv2d_9_pointwise_batchnorm_readvariableop_resource:	?c
Ttrain_mobilenetv1_mobilenetv1_conv2d_9_pointwise_batchnorm_readvariableop_1_resource:	?r
ctrain_mobilenetv1_mobilenetv1_conv2d_9_pointwise_batchnorm_fusedbatchnormv3_readvariableop_resource:	?t
etrain_mobilenetv1_mobilenetv1_conv2d_9_pointwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource:	?n
Strain_mobilenetv1_mobilenetv1_conv2d_10_depthwise_depthwise_readvariableop_resource:?b
Strain_mobilenetv1_mobilenetv1_conv2d_10_depthwise_batchnorm_readvariableop_resource:	?d
Utrain_mobilenetv1_mobilenetv1_conv2d_10_depthwise_batchnorm_readvariableop_1_resource:	?s
dtrain_mobilenetv1_mobilenetv1_conv2d_10_depthwise_batchnorm_fusedbatchnormv3_readvariableop_resource:	?u
ftrain_mobilenetv1_mobilenetv1_conv2d_10_depthwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource:	??
rtrain_mobilenetv1_mobilenetv1_conv2d_10_pointwise_kernel_regularizer_l2_regularizer_l2loss_readvariableop_resource:??b
Strain_mobilenetv1_mobilenetv1_conv2d_10_pointwise_batchnorm_readvariableop_resource:	?d
Utrain_mobilenetv1_mobilenetv1_conv2d_10_pointwise_batchnorm_readvariableop_1_resource:	?s
dtrain_mobilenetv1_mobilenetv1_conv2d_10_pointwise_batchnorm_fusedbatchnormv3_readvariableop_resource:	?u
ftrain_mobilenetv1_mobilenetv1_conv2d_10_pointwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource:	?n
Strain_mobilenetv1_mobilenetv1_conv2d_11_depthwise_depthwise_readvariableop_resource:?b
Strain_mobilenetv1_mobilenetv1_conv2d_11_depthwise_batchnorm_readvariableop_resource:	?d
Utrain_mobilenetv1_mobilenetv1_conv2d_11_depthwise_batchnorm_readvariableop_1_resource:	?s
dtrain_mobilenetv1_mobilenetv1_conv2d_11_depthwise_batchnorm_fusedbatchnormv3_readvariableop_resource:	?u
ftrain_mobilenetv1_mobilenetv1_conv2d_11_depthwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource:	??
rtrain_mobilenetv1_mobilenetv1_conv2d_11_pointwise_kernel_regularizer_l2_regularizer_l2loss_readvariableop_resource:??b
Strain_mobilenetv1_mobilenetv1_conv2d_11_pointwise_batchnorm_readvariableop_resource:	?d
Utrain_mobilenetv1_mobilenetv1_conv2d_11_pointwise_batchnorm_readvariableop_1_resource:	?s
dtrain_mobilenetv1_mobilenetv1_conv2d_11_pointwise_batchnorm_fusedbatchnormv3_readvariableop_resource:	?u
ftrain_mobilenetv1_mobilenetv1_conv2d_11_pointwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource:	?n
Strain_mobilenetv1_mobilenetv1_conv2d_12_depthwise_depthwise_readvariableop_resource:?b
Strain_mobilenetv1_mobilenetv1_conv2d_12_depthwise_batchnorm_readvariableop_resource:	?d
Utrain_mobilenetv1_mobilenetv1_conv2d_12_depthwise_batchnorm_readvariableop_1_resource:	?s
dtrain_mobilenetv1_mobilenetv1_conv2d_12_depthwise_batchnorm_fusedbatchnormv3_readvariableop_resource:	?u
ftrain_mobilenetv1_mobilenetv1_conv2d_12_depthwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource:	??
rtrain_mobilenetv1_mobilenetv1_conv2d_12_pointwise_kernel_regularizer_l2_regularizer_l2loss_readvariableop_resource:??b
Strain_mobilenetv1_mobilenetv1_conv2d_12_pointwise_batchnorm_readvariableop_resource:	?d
Utrain_mobilenetv1_mobilenetv1_conv2d_12_pointwise_batchnorm_readvariableop_1_resource:	?s
dtrain_mobilenetv1_mobilenetv1_conv2d_12_pointwise_batchnorm_fusedbatchnormv3_readvariableop_resource:	?u
ftrain_mobilenetv1_mobilenetv1_conv2d_12_pointwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource:	?n
Strain_mobilenetv1_mobilenetv1_conv2d_13_depthwise_depthwise_readvariableop_resource:?b
Strain_mobilenetv1_mobilenetv1_conv2d_13_depthwise_batchnorm_readvariableop_resource:	?d
Utrain_mobilenetv1_mobilenetv1_conv2d_13_depthwise_batchnorm_readvariableop_1_resource:	?s
dtrain_mobilenetv1_mobilenetv1_conv2d_13_depthwise_batchnorm_fusedbatchnormv3_readvariableop_resource:	?u
ftrain_mobilenetv1_mobilenetv1_conv2d_13_depthwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource:	??
rtrain_mobilenetv1_mobilenetv1_conv2d_13_pointwise_kernel_regularizer_l2_regularizer_l2loss_readvariableop_resource:??b
Strain_mobilenetv1_mobilenetv1_conv2d_13_pointwise_batchnorm_readvariableop_resource:	?d
Utrain_mobilenetv1_mobilenetv1_conv2d_13_pointwise_batchnorm_readvariableop_1_resource:	?s
dtrain_mobilenetv1_mobilenetv1_conv2d_13_pointwise_batchnorm_fusedbatchnormv3_readvariableop_resource:	?u
ftrain_mobilenetv1_mobilenetv1_conv2d_13_pointwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource:	??
gtrain_mobilenetv1_logits_conv2d_1c_1x1_kernel_regularizer_l2_regularizer_l2loss_readvariableop_resource:??U
Ftrain_mobilenetv1_logits_conv2d_1c_1x1_biasadd_readvariableop_resource:	?#
train_total_regularization_loss??
Ntrain/MobilenetV1/MobilenetV1/Conv2d_0/kernel/Regularizer/l2_regularizer/scaleConst*
_output_shapes
: *
dtype0*
valueB
 *??'82P
Ntrain/MobilenetV1/MobilenetV1/Conv2d_0/kernel/Regularizer/l2_regularizer/scale?
^train/MobilenetV1/MobilenetV1/Conv2d_0/kernel/Regularizer/l2_regularizer/L2Loss/ReadVariableOpReadVariableOpgtrain_mobilenetv1_mobilenetv1_conv2d_0_kernel_regularizer_l2_regularizer_l2loss_readvariableop_resource*&
_output_shapes
:*
dtype02`
^train/MobilenetV1/MobilenetV1/Conv2d_0/kernel/Regularizer/l2_regularizer/L2Loss/ReadVariableOp?
Otrain/MobilenetV1/MobilenetV1/Conv2d_0/kernel/Regularizer/l2_regularizer/L2LossL2Lossftrain/MobilenetV1/MobilenetV1/Conv2d_0/kernel/Regularizer/l2_regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: 2Q
Otrain/MobilenetV1/MobilenetV1/Conv2d_0/kernel/Regularizer/l2_regularizer/L2Loss?
Htrain/MobilenetV1/MobilenetV1/Conv2d_0/kernel/Regularizer/l2_regularizerMulWtrain/MobilenetV1/MobilenetV1/Conv2d_0/kernel/Regularizer/l2_regularizer/scale:output:0Xtrain/MobilenetV1/MobilenetV1/Conv2d_0/kernel/Regularizer/l2_regularizer/L2Loss:output:0*
T0*
_output_shapes
: 2J
Htrain/MobilenetV1/MobilenetV1/Conv2d_0/kernel/Regularizer/l2_regularizer?
Xtrain/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/kernel/Regularizer/l2_regularizer/scaleConst*
_output_shapes
: *
dtype0*
valueB
 *??'82Z
Xtrain/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/kernel/Regularizer/l2_regularizer/scale?
htrain/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/kernel/Regularizer/l2_regularizer/L2Loss/ReadVariableOpReadVariableOpqtrain_mobilenetv1_mobilenetv1_conv2d_1_pointwise_kernel_regularizer_l2_regularizer_l2loss_readvariableop_resource*&
_output_shapes
:0*
dtype02j
htrain/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/kernel/Regularizer/l2_regularizer/L2Loss/ReadVariableOp?
Ytrain/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/kernel/Regularizer/l2_regularizer/L2LossL2Lossptrain/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/kernel/Regularizer/l2_regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: 2[
Ytrain/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/kernel/Regularizer/l2_regularizer/L2Loss?
Rtrain/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/kernel/Regularizer/l2_regularizerMulatrain/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/kernel/Regularizer/l2_regularizer/scale:output:0btrain/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/kernel/Regularizer/l2_regularizer/L2Loss:output:0*
T0*
_output_shapes
: 2T
Rtrain/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/kernel/Regularizer/l2_regularizer?
Xtrain/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/kernel/Regularizer/l2_regularizer/scaleConst*
_output_shapes
: *
dtype0*
valueB
 *??'82Z
Xtrain/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/kernel/Regularizer/l2_regularizer/scale?
htrain/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/kernel/Regularizer/l2_regularizer/L2Loss/ReadVariableOpReadVariableOpqtrain_mobilenetv1_mobilenetv1_conv2d_2_pointwise_kernel_regularizer_l2_regularizer_l2loss_readvariableop_resource*&
_output_shapes
:0`*
dtype02j
htrain/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/kernel/Regularizer/l2_regularizer/L2Loss/ReadVariableOp?
Ytrain/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/kernel/Regularizer/l2_regularizer/L2LossL2Lossptrain/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/kernel/Regularizer/l2_regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: 2[
Ytrain/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/kernel/Regularizer/l2_regularizer/L2Loss?
Rtrain/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/kernel/Regularizer/l2_regularizerMulatrain/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/kernel/Regularizer/l2_regularizer/scale:output:0btrain/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/kernel/Regularizer/l2_regularizer/L2Loss:output:0*
T0*
_output_shapes
: 2T
Rtrain/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/kernel/Regularizer/l2_regularizer?
Xtrain/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/kernel/Regularizer/l2_regularizer/scaleConst*
_output_shapes
: *
dtype0*
valueB
 *??'82Z
Xtrain/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/kernel/Regularizer/l2_regularizer/scale?
htrain/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/kernel/Regularizer/l2_regularizer/L2Loss/ReadVariableOpReadVariableOpqtrain_mobilenetv1_mobilenetv1_conv2d_3_pointwise_kernel_regularizer_l2_regularizer_l2loss_readvariableop_resource*&
_output_shapes
:``*
dtype02j
htrain/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/kernel/Regularizer/l2_regularizer/L2Loss/ReadVariableOp?
Ytrain/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/kernel/Regularizer/l2_regularizer/L2LossL2Lossptrain/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/kernel/Regularizer/l2_regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: 2[
Ytrain/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/kernel/Regularizer/l2_regularizer/L2Loss?
Rtrain/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/kernel/Regularizer/l2_regularizerMulatrain/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/kernel/Regularizer/l2_regularizer/scale:output:0btrain/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/kernel/Regularizer/l2_regularizer/L2Loss:output:0*
T0*
_output_shapes
: 2T
Rtrain/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/kernel/Regularizer/l2_regularizer?
Xtrain/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/kernel/Regularizer/l2_regularizer/scaleConst*
_output_shapes
: *
dtype0*
valueB
 *??'82Z
Xtrain/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/kernel/Regularizer/l2_regularizer/scale?
htrain/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/kernel/Regularizer/l2_regularizer/L2Loss/ReadVariableOpReadVariableOpqtrain_mobilenetv1_mobilenetv1_conv2d_4_pointwise_kernel_regularizer_l2_regularizer_l2loss_readvariableop_resource*'
_output_shapes
:`?*
dtype02j
htrain/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/kernel/Regularizer/l2_regularizer/L2Loss/ReadVariableOp?
Ytrain/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/kernel/Regularizer/l2_regularizer/L2LossL2Lossptrain/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/kernel/Regularizer/l2_regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: 2[
Ytrain/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/kernel/Regularizer/l2_regularizer/L2Loss?
Rtrain/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/kernel/Regularizer/l2_regularizerMulatrain/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/kernel/Regularizer/l2_regularizer/scale:output:0btrain/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/kernel/Regularizer/l2_regularizer/L2Loss:output:0*
T0*
_output_shapes
: 2T
Rtrain/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/kernel/Regularizer/l2_regularizer?
Xtrain/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/kernel/Regularizer/l2_regularizer/scaleConst*
_output_shapes
: *
dtype0*
valueB
 *??'82Z
Xtrain/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/kernel/Regularizer/l2_regularizer/scale?
htrain/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/kernel/Regularizer/l2_regularizer/L2Loss/ReadVariableOpReadVariableOpqtrain_mobilenetv1_mobilenetv1_conv2d_5_pointwise_kernel_regularizer_l2_regularizer_l2loss_readvariableop_resource*(
_output_shapes
:??*
dtype02j
htrain/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/kernel/Regularizer/l2_regularizer/L2Loss/ReadVariableOp?
Ytrain/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/kernel/Regularizer/l2_regularizer/L2LossL2Lossptrain/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/kernel/Regularizer/l2_regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: 2[
Ytrain/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/kernel/Regularizer/l2_regularizer/L2Loss?
Rtrain/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/kernel/Regularizer/l2_regularizerMulatrain/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/kernel/Regularizer/l2_regularizer/scale:output:0btrain/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/kernel/Regularizer/l2_regularizer/L2Loss:output:0*
T0*
_output_shapes
: 2T
Rtrain/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/kernel/Regularizer/l2_regularizer?
Xtrain/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/kernel/Regularizer/l2_regularizer/scaleConst*
_output_shapes
: *
dtype0*
valueB
 *??'82Z
Xtrain/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/kernel/Regularizer/l2_regularizer/scale?
htrain/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/kernel/Regularizer/l2_regularizer/L2Loss/ReadVariableOpReadVariableOpqtrain_mobilenetv1_mobilenetv1_conv2d_6_pointwise_kernel_regularizer_l2_regularizer_l2loss_readvariableop_resource*(
_output_shapes
:??*
dtype02j
htrain/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/kernel/Regularizer/l2_regularizer/L2Loss/ReadVariableOp?
Ytrain/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/kernel/Regularizer/l2_regularizer/L2LossL2Lossptrain/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/kernel/Regularizer/l2_regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: 2[
Ytrain/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/kernel/Regularizer/l2_regularizer/L2Loss?
Rtrain/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/kernel/Regularizer/l2_regularizerMulatrain/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/kernel/Regularizer/l2_regularizer/scale:output:0btrain/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/kernel/Regularizer/l2_regularizer/L2Loss:output:0*
T0*
_output_shapes
: 2T
Rtrain/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/kernel/Regularizer/l2_regularizer?
Xtrain/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/kernel/Regularizer/l2_regularizer/scaleConst*
_output_shapes
: *
dtype0*
valueB
 *??'82Z
Xtrain/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/kernel/Regularizer/l2_regularizer/scale?
htrain/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/kernel/Regularizer/l2_regularizer/L2Loss/ReadVariableOpReadVariableOpqtrain_mobilenetv1_mobilenetv1_conv2d_7_pointwise_kernel_regularizer_l2_regularizer_l2loss_readvariableop_resource*(
_output_shapes
:??*
dtype02j
htrain/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/kernel/Regularizer/l2_regularizer/L2Loss/ReadVariableOp?
Ytrain/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/kernel/Regularizer/l2_regularizer/L2LossL2Lossptrain/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/kernel/Regularizer/l2_regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: 2[
Ytrain/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/kernel/Regularizer/l2_regularizer/L2Loss?
Rtrain/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/kernel/Regularizer/l2_regularizerMulatrain/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/kernel/Regularizer/l2_regularizer/scale:output:0btrain/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/kernel/Regularizer/l2_regularizer/L2Loss:output:0*
T0*
_output_shapes
: 2T
Rtrain/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/kernel/Regularizer/l2_regularizer?
Xtrain/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/kernel/Regularizer/l2_regularizer/scaleConst*
_output_shapes
: *
dtype0*
valueB
 *??'82Z
Xtrain/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/kernel/Regularizer/l2_regularizer/scale?
htrain/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/kernel/Regularizer/l2_regularizer/L2Loss/ReadVariableOpReadVariableOpqtrain_mobilenetv1_mobilenetv1_conv2d_8_pointwise_kernel_regularizer_l2_regularizer_l2loss_readvariableop_resource*(
_output_shapes
:??*
dtype02j
htrain/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/kernel/Regularizer/l2_regularizer/L2Loss/ReadVariableOp?
Ytrain/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/kernel/Regularizer/l2_regularizer/L2LossL2Lossptrain/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/kernel/Regularizer/l2_regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: 2[
Ytrain/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/kernel/Regularizer/l2_regularizer/L2Loss?
Rtrain/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/kernel/Regularizer/l2_regularizerMulatrain/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/kernel/Regularizer/l2_regularizer/scale:output:0btrain/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/kernel/Regularizer/l2_regularizer/L2Loss:output:0*
T0*
_output_shapes
: 2T
Rtrain/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/kernel/Regularizer/l2_regularizer?
Xtrain/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/kernel/Regularizer/l2_regularizer/scaleConst*
_output_shapes
: *
dtype0*
valueB
 *??'82Z
Xtrain/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/kernel/Regularizer/l2_regularizer/scale?
htrain/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/kernel/Regularizer/l2_regularizer/L2Loss/ReadVariableOpReadVariableOpqtrain_mobilenetv1_mobilenetv1_conv2d_9_pointwise_kernel_regularizer_l2_regularizer_l2loss_readvariableop_resource*(
_output_shapes
:??*
dtype02j
htrain/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/kernel/Regularizer/l2_regularizer/L2Loss/ReadVariableOp?
Ytrain/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/kernel/Regularizer/l2_regularizer/L2LossL2Lossptrain/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/kernel/Regularizer/l2_regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: 2[
Ytrain/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/kernel/Regularizer/l2_regularizer/L2Loss?
Rtrain/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/kernel/Regularizer/l2_regularizerMulatrain/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/kernel/Regularizer/l2_regularizer/scale:output:0btrain/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/kernel/Regularizer/l2_regularizer/L2Loss:output:0*
T0*
_output_shapes
: 2T
Rtrain/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/kernel/Regularizer/l2_regularizer?
Ytrain/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/kernel/Regularizer/l2_regularizer/scaleConst*
_output_shapes
: *
dtype0*
valueB
 *??'82[
Ytrain/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/kernel/Regularizer/l2_regularizer/scale?
itrain/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/kernel/Regularizer/l2_regularizer/L2Loss/ReadVariableOpReadVariableOprtrain_mobilenetv1_mobilenetv1_conv2d_10_pointwise_kernel_regularizer_l2_regularizer_l2loss_readvariableop_resource*(
_output_shapes
:??*
dtype02k
itrain/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/kernel/Regularizer/l2_regularizer/L2Loss/ReadVariableOp?
Ztrain/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/kernel/Regularizer/l2_regularizer/L2LossL2Lossqtrain/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/kernel/Regularizer/l2_regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: 2\
Ztrain/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/kernel/Regularizer/l2_regularizer/L2Loss?
Strain/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/kernel/Regularizer/l2_regularizerMulbtrain/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/kernel/Regularizer/l2_regularizer/scale:output:0ctrain/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/kernel/Regularizer/l2_regularizer/L2Loss:output:0*
T0*
_output_shapes
: 2U
Strain/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/kernel/Regularizer/l2_regularizer?
Ytrain/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/kernel/Regularizer/l2_regularizer/scaleConst*
_output_shapes
: *
dtype0*
valueB
 *??'82[
Ytrain/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/kernel/Regularizer/l2_regularizer/scale?
itrain/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/kernel/Regularizer/l2_regularizer/L2Loss/ReadVariableOpReadVariableOprtrain_mobilenetv1_mobilenetv1_conv2d_11_pointwise_kernel_regularizer_l2_regularizer_l2loss_readvariableop_resource*(
_output_shapes
:??*
dtype02k
itrain/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/kernel/Regularizer/l2_regularizer/L2Loss/ReadVariableOp?
Ztrain/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/kernel/Regularizer/l2_regularizer/L2LossL2Lossqtrain/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/kernel/Regularizer/l2_regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: 2\
Ztrain/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/kernel/Regularizer/l2_regularizer/L2Loss?
Strain/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/kernel/Regularizer/l2_regularizerMulbtrain/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/kernel/Regularizer/l2_regularizer/scale:output:0ctrain/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/kernel/Regularizer/l2_regularizer/L2Loss:output:0*
T0*
_output_shapes
: 2U
Strain/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/kernel/Regularizer/l2_regularizer?
Ytrain/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/kernel/Regularizer/l2_regularizer/scaleConst*
_output_shapes
: *
dtype0*
valueB
 *??'82[
Ytrain/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/kernel/Regularizer/l2_regularizer/scale?
itrain/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/kernel/Regularizer/l2_regularizer/L2Loss/ReadVariableOpReadVariableOprtrain_mobilenetv1_mobilenetv1_conv2d_12_pointwise_kernel_regularizer_l2_regularizer_l2loss_readvariableop_resource*(
_output_shapes
:??*
dtype02k
itrain/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/kernel/Regularizer/l2_regularizer/L2Loss/ReadVariableOp?
Ztrain/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/kernel/Regularizer/l2_regularizer/L2LossL2Lossqtrain/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/kernel/Regularizer/l2_regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: 2\
Ztrain/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/kernel/Regularizer/l2_regularizer/L2Loss?
Strain/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/kernel/Regularizer/l2_regularizerMulbtrain/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/kernel/Regularizer/l2_regularizer/scale:output:0ctrain/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/kernel/Regularizer/l2_regularizer/L2Loss:output:0*
T0*
_output_shapes
: 2U
Strain/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/kernel/Regularizer/l2_regularizer?
Ytrain/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/kernel/Regularizer/l2_regularizer/scaleConst*
_output_shapes
: *
dtype0*
valueB
 *??'82[
Ytrain/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/kernel/Regularizer/l2_regularizer/scale?
itrain/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/kernel/Regularizer/l2_regularizer/L2Loss/ReadVariableOpReadVariableOprtrain_mobilenetv1_mobilenetv1_conv2d_13_pointwise_kernel_regularizer_l2_regularizer_l2loss_readvariableop_resource*(
_output_shapes
:??*
dtype02k
itrain/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/kernel/Regularizer/l2_regularizer/L2Loss/ReadVariableOp?
Ztrain/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/kernel/Regularizer/l2_regularizer/L2LossL2Lossqtrain/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/kernel/Regularizer/l2_regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: 2\
Ztrain/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/kernel/Regularizer/l2_regularizer/L2Loss?
Strain/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/kernel/Regularizer/l2_regularizerMulbtrain/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/kernel/Regularizer/l2_regularizer/scale:output:0ctrain/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/kernel/Regularizer/l2_regularizer/L2Loss:output:0*
T0*
_output_shapes
: 2U
Strain/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/kernel/Regularizer/l2_regularizer?
Ntrain/MobilenetV1/Logits/Conv2d_1c_1x1/kernel/Regularizer/l2_regularizer/scaleConst*
_output_shapes
: *
dtype0*
valueB
 *??'82P
Ntrain/MobilenetV1/Logits/Conv2d_1c_1x1/kernel/Regularizer/l2_regularizer/scale?
^train/MobilenetV1/Logits/Conv2d_1c_1x1/kernel/Regularizer/l2_regularizer/L2Loss/ReadVariableOpReadVariableOpgtrain_mobilenetv1_logits_conv2d_1c_1x1_kernel_regularizer_l2_regularizer_l2loss_readvariableop_resource*(
_output_shapes
:??*
dtype02`
^train/MobilenetV1/Logits/Conv2d_1c_1x1/kernel/Regularizer/l2_regularizer/L2Loss/ReadVariableOp?
Otrain/MobilenetV1/Logits/Conv2d_1c_1x1/kernel/Regularizer/l2_regularizer/L2LossL2Lossftrain/MobilenetV1/Logits/Conv2d_1c_1x1/kernel/Regularizer/l2_regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: 2Q
Otrain/MobilenetV1/Logits/Conv2d_1c_1x1/kernel/Regularizer/l2_regularizer/L2Loss?
Htrain/MobilenetV1/Logits/Conv2d_1c_1x1/kernel/Regularizer/l2_regularizerMulWtrain/MobilenetV1/Logits/Conv2d_1c_1x1/kernel/Regularizer/l2_regularizer/scale:output:0Xtrain/MobilenetV1/Logits/Conv2d_1c_1x1/kernel/Regularizer/l2_regularizer/L2Loss:output:0*
T0*
_output_shapes
: 2J
Htrain/MobilenetV1/Logits/Conv2d_1c_1x1/kernel/Regularizer/l2_regularizer?
train/total_regularization_lossAddNLtrain/MobilenetV1/MobilenetV1/Conv2d_0/kernel/Regularizer/l2_regularizer:z:0Vtrain/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/kernel/Regularizer/l2_regularizer:z:0Vtrain/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/kernel/Regularizer/l2_regularizer:z:0Vtrain/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/kernel/Regularizer/l2_regularizer:z:0Vtrain/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/kernel/Regularizer/l2_regularizer:z:0Vtrain/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/kernel/Regularizer/l2_regularizer:z:0Vtrain/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/kernel/Regularizer/l2_regularizer:z:0Vtrain/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/kernel/Regularizer/l2_regularizer:z:0Vtrain/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/kernel/Regularizer/l2_regularizer:z:0Vtrain/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/kernel/Regularizer/l2_regularizer:z:0Wtrain/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/kernel/Regularizer/l2_regularizer:z:0Wtrain/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/kernel/Regularizer/l2_regularizer:z:0Wtrain/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/kernel/Regularizer/l2_regularizer:z:0Wtrain/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/kernel/Regularizer/l2_regularizer:z:0Ltrain/MobilenetV1/Logits/Conv2d_1c_1x1/kernel/Regularizer/l2_regularizer:z:0*
N*
T0*
_output_shapes
: 2!
train/total_regularization_loss"H
train_total_regularization_loss%train/total_regularization_loss:sum:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 
?8
? 
"__inference_signature_wrapper_5765

inputs!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:#
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:#
	unknown_9:0

unknown_10:0

unknown_11:0

unknown_12:0

unknown_13:0$

unknown_14:0

unknown_15:0

unknown_16:0

unknown_17:0

unknown_18:0$

unknown_19:0`

unknown_20:`

unknown_21:`

unknown_22:`

unknown_23:`$

unknown_24:`

unknown_25:`

unknown_26:`

unknown_27:`

unknown_28:`$

unknown_29:``

unknown_30:`

unknown_31:`

unknown_32:`

unknown_33:`$

unknown_34:`

unknown_35:`

unknown_36:`

unknown_37:`

unknown_38:`%

unknown_39:`?

unknown_40:	?

unknown_41:	?

unknown_42:	?

unknown_43:	?%

unknown_44:?

unknown_45:	?

unknown_46:	?

unknown_47:	?

unknown_48:	?&

unknown_49:??

unknown_50:	?

unknown_51:	?

unknown_52:	?

unknown_53:	?%

unknown_54:?

unknown_55:	?

unknown_56:	?

unknown_57:	?

unknown_58:	?&

unknown_59:??

unknown_60:	?

unknown_61:	?

unknown_62:	?

unknown_63:	?%

unknown_64:?

unknown_65:	?

unknown_66:	?

unknown_67:	?

unknown_68:	?&

unknown_69:??

unknown_70:	?

unknown_71:	?

unknown_72:	?

unknown_73:	?%

unknown_74:?

unknown_75:	?

unknown_76:	?

unknown_77:	?

unknown_78:	?&

unknown_79:??

unknown_80:	?

unknown_81:	?

unknown_82:	?

unknown_83:	?%

unknown_84:?

unknown_85:	?

unknown_86:	?

unknown_87:	?

unknown_88:	?&

unknown_89:??

unknown_90:	?

unknown_91:	?

unknown_92:	?

unknown_93:	?%

unknown_94:?

unknown_95:	?

unknown_96:	?

unknown_97:	?

unknown_98:	?&

unknown_99:??
unknown_100:	?
unknown_101:	?
unknown_102:	?
unknown_103:	?&
unknown_104:?
unknown_105:	?
unknown_106:	?
unknown_107:	?
unknown_108:	?'
unknown_109:??
unknown_110:	?
unknown_111:	?
unknown_112:	?
unknown_113:	?&
unknown_114:?
unknown_115:	?
unknown_116:	?
unknown_117:	?
unknown_118:	?'
unknown_119:??
unknown_120:	?
unknown_121:	?
unknown_122:	?
unknown_123:	?&
unknown_124:?
unknown_125:	?
unknown_126:	?
unknown_127:	?
unknown_128:	?'
unknown_129:??
unknown_130:	?
unknown_131:	?
unknown_132:	?
unknown_133:	?'
unknown_134:??
unknown_135:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58
unknown_59
unknown_60
unknown_61
unknown_62
unknown_63
unknown_64
unknown_65
unknown_66
unknown_67
unknown_68
unknown_69
unknown_70
unknown_71
unknown_72
unknown_73
unknown_74
unknown_75
unknown_76
unknown_77
unknown_78
unknown_79
unknown_80
unknown_81
unknown_82
unknown_83
unknown_84
unknown_85
unknown_86
unknown_87
unknown_88
unknown_89
unknown_90
unknown_91
unknown_92
unknown_93
unknown_94
unknown_95
unknown_96
unknown_97
unknown_98
unknown_99unknown_100unknown_101unknown_102unknown_103unknown_104unknown_105unknown_106unknown_107unknown_108unknown_109unknown_110unknown_111unknown_112unknown_113unknown_114unknown_115unknown_116unknown_117unknown_118unknown_119unknown_120unknown_121unknown_122unknown_123unknown_124unknown_125unknown_126unknown_127unknown_128unknown_129unknown_130unknown_131unknown_132unknown_133unknown_134unknown_135*?
Tin?
?2?*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*?
_read_only_resource_inputs?
??	
 !"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~??????????*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference_serving_default_fn_54842
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

IdentityD
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:+???????????????????????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
??
??
__inference_pruned_3393
placeholder_1
placeholder_2?
gtrain_mobilenetv1_mobilenetv1_conv2d_0_kernel_regularizer_l2_regularizer_l2loss_readvariableop_resource:V
Htrain_mobilenetv1_mobilenetv1_conv2d_0_batchnorm_readvariableop_resource:X
Jtrain_mobilenetv1_mobilenetv1_conv2d_0_batchnorm_readvariableop_1_resource:g
Ytrain_mobilenetv1_mobilenetv1_conv2d_0_batchnorm_fusedbatchnormv3_readvariableop_resource:i
[train_mobilenetv1_mobilenetv1_conv2d_0_batchnorm_fusedbatchnormv3_readvariableop_1_resource:l
Rtrain_mobilenetv1_mobilenetv1_conv2d_1_depthwise_depthwise_readvariableop_resource:`
Rtrain_mobilenetv1_mobilenetv1_conv2d_1_depthwise_batchnorm_readvariableop_resource:b
Ttrain_mobilenetv1_mobilenetv1_conv2d_1_depthwise_batchnorm_readvariableop_1_resource:q
ctrain_mobilenetv1_mobilenetv1_conv2d_1_depthwise_batchnorm_fusedbatchnormv3_readvariableop_resource:s
etrain_mobilenetv1_mobilenetv1_conv2d_1_depthwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource:?
qtrain_mobilenetv1_mobilenetv1_conv2d_1_pointwise_kernel_regularizer_l2_regularizer_l2loss_readvariableop_resource:0`
Rtrain_mobilenetv1_mobilenetv1_conv2d_1_pointwise_batchnorm_readvariableop_resource:0b
Ttrain_mobilenetv1_mobilenetv1_conv2d_1_pointwise_batchnorm_readvariableop_1_resource:0q
ctrain_mobilenetv1_mobilenetv1_conv2d_1_pointwise_batchnorm_fusedbatchnormv3_readvariableop_resource:0s
etrain_mobilenetv1_mobilenetv1_conv2d_1_pointwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource:0l
Rtrain_mobilenetv1_mobilenetv1_conv2d_2_depthwise_depthwise_readvariableop_resource:0`
Rtrain_mobilenetv1_mobilenetv1_conv2d_2_depthwise_batchnorm_readvariableop_resource:0b
Ttrain_mobilenetv1_mobilenetv1_conv2d_2_depthwise_batchnorm_readvariableop_1_resource:0q
ctrain_mobilenetv1_mobilenetv1_conv2d_2_depthwise_batchnorm_fusedbatchnormv3_readvariableop_resource:0s
etrain_mobilenetv1_mobilenetv1_conv2d_2_depthwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource:0?
qtrain_mobilenetv1_mobilenetv1_conv2d_2_pointwise_kernel_regularizer_l2_regularizer_l2loss_readvariableop_resource:0``
Rtrain_mobilenetv1_mobilenetv1_conv2d_2_pointwise_batchnorm_readvariableop_resource:`b
Ttrain_mobilenetv1_mobilenetv1_conv2d_2_pointwise_batchnorm_readvariableop_1_resource:`q
ctrain_mobilenetv1_mobilenetv1_conv2d_2_pointwise_batchnorm_fusedbatchnormv3_readvariableop_resource:`s
etrain_mobilenetv1_mobilenetv1_conv2d_2_pointwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource:`l
Rtrain_mobilenetv1_mobilenetv1_conv2d_3_depthwise_depthwise_readvariableop_resource:``
Rtrain_mobilenetv1_mobilenetv1_conv2d_3_depthwise_batchnorm_readvariableop_resource:`b
Ttrain_mobilenetv1_mobilenetv1_conv2d_3_depthwise_batchnorm_readvariableop_1_resource:`q
ctrain_mobilenetv1_mobilenetv1_conv2d_3_depthwise_batchnorm_fusedbatchnormv3_readvariableop_resource:`s
etrain_mobilenetv1_mobilenetv1_conv2d_3_depthwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource:`?
qtrain_mobilenetv1_mobilenetv1_conv2d_3_pointwise_kernel_regularizer_l2_regularizer_l2loss_readvariableop_resource:```
Rtrain_mobilenetv1_mobilenetv1_conv2d_3_pointwise_batchnorm_readvariableop_resource:`b
Ttrain_mobilenetv1_mobilenetv1_conv2d_3_pointwise_batchnorm_readvariableop_1_resource:`q
ctrain_mobilenetv1_mobilenetv1_conv2d_3_pointwise_batchnorm_fusedbatchnormv3_readvariableop_resource:`s
etrain_mobilenetv1_mobilenetv1_conv2d_3_pointwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource:`l
Rtrain_mobilenetv1_mobilenetv1_conv2d_4_depthwise_depthwise_readvariableop_resource:``
Rtrain_mobilenetv1_mobilenetv1_conv2d_4_depthwise_batchnorm_readvariableop_resource:`b
Ttrain_mobilenetv1_mobilenetv1_conv2d_4_depthwise_batchnorm_readvariableop_1_resource:`q
ctrain_mobilenetv1_mobilenetv1_conv2d_4_depthwise_batchnorm_fusedbatchnormv3_readvariableop_resource:`s
etrain_mobilenetv1_mobilenetv1_conv2d_4_depthwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource:`?
qtrain_mobilenetv1_mobilenetv1_conv2d_4_pointwise_kernel_regularizer_l2_regularizer_l2loss_readvariableop_resource:`?a
Rtrain_mobilenetv1_mobilenetv1_conv2d_4_pointwise_batchnorm_readvariableop_resource:	?c
Ttrain_mobilenetv1_mobilenetv1_conv2d_4_pointwise_batchnorm_readvariableop_1_resource:	?r
ctrain_mobilenetv1_mobilenetv1_conv2d_4_pointwise_batchnorm_fusedbatchnormv3_readvariableop_resource:	?t
etrain_mobilenetv1_mobilenetv1_conv2d_4_pointwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource:	?m
Rtrain_mobilenetv1_mobilenetv1_conv2d_5_depthwise_depthwise_readvariableop_resource:?a
Rtrain_mobilenetv1_mobilenetv1_conv2d_5_depthwise_batchnorm_readvariableop_resource:	?c
Ttrain_mobilenetv1_mobilenetv1_conv2d_5_depthwise_batchnorm_readvariableop_1_resource:	?r
ctrain_mobilenetv1_mobilenetv1_conv2d_5_depthwise_batchnorm_fusedbatchnormv3_readvariableop_resource:	?t
etrain_mobilenetv1_mobilenetv1_conv2d_5_depthwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource:	??
qtrain_mobilenetv1_mobilenetv1_conv2d_5_pointwise_kernel_regularizer_l2_regularizer_l2loss_readvariableop_resource:??a
Rtrain_mobilenetv1_mobilenetv1_conv2d_5_pointwise_batchnorm_readvariableop_resource:	?c
Ttrain_mobilenetv1_mobilenetv1_conv2d_5_pointwise_batchnorm_readvariableop_1_resource:	?r
ctrain_mobilenetv1_mobilenetv1_conv2d_5_pointwise_batchnorm_fusedbatchnormv3_readvariableop_resource:	?t
etrain_mobilenetv1_mobilenetv1_conv2d_5_pointwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource:	?m
Rtrain_mobilenetv1_mobilenetv1_conv2d_6_depthwise_depthwise_readvariableop_resource:?a
Rtrain_mobilenetv1_mobilenetv1_conv2d_6_depthwise_batchnorm_readvariableop_resource:	?c
Ttrain_mobilenetv1_mobilenetv1_conv2d_6_depthwise_batchnorm_readvariableop_1_resource:	?r
ctrain_mobilenetv1_mobilenetv1_conv2d_6_depthwise_batchnorm_fusedbatchnormv3_readvariableop_resource:	?t
etrain_mobilenetv1_mobilenetv1_conv2d_6_depthwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource:	??
qtrain_mobilenetv1_mobilenetv1_conv2d_6_pointwise_kernel_regularizer_l2_regularizer_l2loss_readvariableop_resource:??a
Rtrain_mobilenetv1_mobilenetv1_conv2d_6_pointwise_batchnorm_readvariableop_resource:	?c
Ttrain_mobilenetv1_mobilenetv1_conv2d_6_pointwise_batchnorm_readvariableop_1_resource:	?r
ctrain_mobilenetv1_mobilenetv1_conv2d_6_pointwise_batchnorm_fusedbatchnormv3_readvariableop_resource:	?t
etrain_mobilenetv1_mobilenetv1_conv2d_6_pointwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource:	?m
Rtrain_mobilenetv1_mobilenetv1_conv2d_7_depthwise_depthwise_readvariableop_resource:?a
Rtrain_mobilenetv1_mobilenetv1_conv2d_7_depthwise_batchnorm_readvariableop_resource:	?c
Ttrain_mobilenetv1_mobilenetv1_conv2d_7_depthwise_batchnorm_readvariableop_1_resource:	?r
ctrain_mobilenetv1_mobilenetv1_conv2d_7_depthwise_batchnorm_fusedbatchnormv3_readvariableop_resource:	?t
etrain_mobilenetv1_mobilenetv1_conv2d_7_depthwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource:	??
qtrain_mobilenetv1_mobilenetv1_conv2d_7_pointwise_kernel_regularizer_l2_regularizer_l2loss_readvariableop_resource:??a
Rtrain_mobilenetv1_mobilenetv1_conv2d_7_pointwise_batchnorm_readvariableop_resource:	?c
Ttrain_mobilenetv1_mobilenetv1_conv2d_7_pointwise_batchnorm_readvariableop_1_resource:	?r
ctrain_mobilenetv1_mobilenetv1_conv2d_7_pointwise_batchnorm_fusedbatchnormv3_readvariableop_resource:	?t
etrain_mobilenetv1_mobilenetv1_conv2d_7_pointwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource:	?m
Rtrain_mobilenetv1_mobilenetv1_conv2d_8_depthwise_depthwise_readvariableop_resource:?a
Rtrain_mobilenetv1_mobilenetv1_conv2d_8_depthwise_batchnorm_readvariableop_resource:	?c
Ttrain_mobilenetv1_mobilenetv1_conv2d_8_depthwise_batchnorm_readvariableop_1_resource:	?r
ctrain_mobilenetv1_mobilenetv1_conv2d_8_depthwise_batchnorm_fusedbatchnormv3_readvariableop_resource:	?t
etrain_mobilenetv1_mobilenetv1_conv2d_8_depthwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource:	??
qtrain_mobilenetv1_mobilenetv1_conv2d_8_pointwise_kernel_regularizer_l2_regularizer_l2loss_readvariableop_resource:??a
Rtrain_mobilenetv1_mobilenetv1_conv2d_8_pointwise_batchnorm_readvariableop_resource:	?c
Ttrain_mobilenetv1_mobilenetv1_conv2d_8_pointwise_batchnorm_readvariableop_1_resource:	?r
ctrain_mobilenetv1_mobilenetv1_conv2d_8_pointwise_batchnorm_fusedbatchnormv3_readvariableop_resource:	?t
etrain_mobilenetv1_mobilenetv1_conv2d_8_pointwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource:	?m
Rtrain_mobilenetv1_mobilenetv1_conv2d_9_depthwise_depthwise_readvariableop_resource:?a
Rtrain_mobilenetv1_mobilenetv1_conv2d_9_depthwise_batchnorm_readvariableop_resource:	?c
Ttrain_mobilenetv1_mobilenetv1_conv2d_9_depthwise_batchnorm_readvariableop_1_resource:	?r
ctrain_mobilenetv1_mobilenetv1_conv2d_9_depthwise_batchnorm_fusedbatchnormv3_readvariableop_resource:	?t
etrain_mobilenetv1_mobilenetv1_conv2d_9_depthwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource:	??
qtrain_mobilenetv1_mobilenetv1_conv2d_9_pointwise_kernel_regularizer_l2_regularizer_l2loss_readvariableop_resource:??a
Rtrain_mobilenetv1_mobilenetv1_conv2d_9_pointwise_batchnorm_readvariableop_resource:	?c
Ttrain_mobilenetv1_mobilenetv1_conv2d_9_pointwise_batchnorm_readvariableop_1_resource:	?r
ctrain_mobilenetv1_mobilenetv1_conv2d_9_pointwise_batchnorm_fusedbatchnormv3_readvariableop_resource:	?t
etrain_mobilenetv1_mobilenetv1_conv2d_9_pointwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource:	?n
Strain_mobilenetv1_mobilenetv1_conv2d_10_depthwise_depthwise_readvariableop_resource:?b
Strain_mobilenetv1_mobilenetv1_conv2d_10_depthwise_batchnorm_readvariableop_resource:	?d
Utrain_mobilenetv1_mobilenetv1_conv2d_10_depthwise_batchnorm_readvariableop_1_resource:	?s
dtrain_mobilenetv1_mobilenetv1_conv2d_10_depthwise_batchnorm_fusedbatchnormv3_readvariableop_resource:	?u
ftrain_mobilenetv1_mobilenetv1_conv2d_10_depthwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource:	??
rtrain_mobilenetv1_mobilenetv1_conv2d_10_pointwise_kernel_regularizer_l2_regularizer_l2loss_readvariableop_resource:??b
Strain_mobilenetv1_mobilenetv1_conv2d_10_pointwise_batchnorm_readvariableop_resource:	?d
Utrain_mobilenetv1_mobilenetv1_conv2d_10_pointwise_batchnorm_readvariableop_1_resource:	?s
dtrain_mobilenetv1_mobilenetv1_conv2d_10_pointwise_batchnorm_fusedbatchnormv3_readvariableop_resource:	?u
ftrain_mobilenetv1_mobilenetv1_conv2d_10_pointwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource:	?n
Strain_mobilenetv1_mobilenetv1_conv2d_11_depthwise_depthwise_readvariableop_resource:?b
Strain_mobilenetv1_mobilenetv1_conv2d_11_depthwise_batchnorm_readvariableop_resource:	?d
Utrain_mobilenetv1_mobilenetv1_conv2d_11_depthwise_batchnorm_readvariableop_1_resource:	?s
dtrain_mobilenetv1_mobilenetv1_conv2d_11_depthwise_batchnorm_fusedbatchnormv3_readvariableop_resource:	?u
ftrain_mobilenetv1_mobilenetv1_conv2d_11_depthwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource:	??
rtrain_mobilenetv1_mobilenetv1_conv2d_11_pointwise_kernel_regularizer_l2_regularizer_l2loss_readvariableop_resource:??b
Strain_mobilenetv1_mobilenetv1_conv2d_11_pointwise_batchnorm_readvariableop_resource:	?d
Utrain_mobilenetv1_mobilenetv1_conv2d_11_pointwise_batchnorm_readvariableop_1_resource:	?s
dtrain_mobilenetv1_mobilenetv1_conv2d_11_pointwise_batchnorm_fusedbatchnormv3_readvariableop_resource:	?u
ftrain_mobilenetv1_mobilenetv1_conv2d_11_pointwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource:	?n
Strain_mobilenetv1_mobilenetv1_conv2d_12_depthwise_depthwise_readvariableop_resource:?b
Strain_mobilenetv1_mobilenetv1_conv2d_12_depthwise_batchnorm_readvariableop_resource:	?d
Utrain_mobilenetv1_mobilenetv1_conv2d_12_depthwise_batchnorm_readvariableop_1_resource:	?s
dtrain_mobilenetv1_mobilenetv1_conv2d_12_depthwise_batchnorm_fusedbatchnormv3_readvariableop_resource:	?u
ftrain_mobilenetv1_mobilenetv1_conv2d_12_depthwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource:	??
rtrain_mobilenetv1_mobilenetv1_conv2d_12_pointwise_kernel_regularizer_l2_regularizer_l2loss_readvariableop_resource:??b
Strain_mobilenetv1_mobilenetv1_conv2d_12_pointwise_batchnorm_readvariableop_resource:	?d
Utrain_mobilenetv1_mobilenetv1_conv2d_12_pointwise_batchnorm_readvariableop_1_resource:	?s
dtrain_mobilenetv1_mobilenetv1_conv2d_12_pointwise_batchnorm_fusedbatchnormv3_readvariableop_resource:	?u
ftrain_mobilenetv1_mobilenetv1_conv2d_12_pointwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource:	?n
Strain_mobilenetv1_mobilenetv1_conv2d_13_depthwise_depthwise_readvariableop_resource:?b
Strain_mobilenetv1_mobilenetv1_conv2d_13_depthwise_batchnorm_readvariableop_resource:	?d
Utrain_mobilenetv1_mobilenetv1_conv2d_13_depthwise_batchnorm_readvariableop_1_resource:	?s
dtrain_mobilenetv1_mobilenetv1_conv2d_13_depthwise_batchnorm_fusedbatchnormv3_readvariableop_resource:	?u
ftrain_mobilenetv1_mobilenetv1_conv2d_13_depthwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource:	??
rtrain_mobilenetv1_mobilenetv1_conv2d_13_pointwise_kernel_regularizer_l2_regularizer_l2loss_readvariableop_resource:??b
Strain_mobilenetv1_mobilenetv1_conv2d_13_pointwise_batchnorm_readvariableop_resource:	?d
Utrain_mobilenetv1_mobilenetv1_conv2d_13_pointwise_batchnorm_readvariableop_1_resource:	?s
dtrain_mobilenetv1_mobilenetv1_conv2d_13_pointwise_batchnorm_fusedbatchnormv3_readvariableop_resource:	?u
ftrain_mobilenetv1_mobilenetv1_conv2d_13_pointwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource:	??
gtrain_mobilenetv1_logits_conv2d_1c_1x1_kernel_regularizer_l2_regularizer_l2loss_readvariableop_resource:??U
Ftrain_mobilenetv1_logits_conv2d_1c_1x1_biasadd_readvariableop_resource:	?-
)predict_mobilenetv1_logits_spatialsqueeze2
.predict_mobilenetv1_mobilenetv1_conv2d_0_relu6=
9predict_mobilenetv1_mobilenetv1_conv2d_10_depthwise_relu6=
9predict_mobilenetv1_mobilenetv1_conv2d_10_pointwise_relu6=
9predict_mobilenetv1_mobilenetv1_conv2d_11_depthwise_relu6=
9predict_mobilenetv1_mobilenetv1_conv2d_11_pointwise_relu6=
9predict_mobilenetv1_mobilenetv1_conv2d_12_depthwise_relu6=
9predict_mobilenetv1_mobilenetv1_conv2d_12_pointwise_relu6=
9predict_mobilenetv1_mobilenetv1_conv2d_13_depthwise_relu6=
9predict_mobilenetv1_mobilenetv1_conv2d_13_pointwise_relu6<
8predict_mobilenetv1_mobilenetv1_conv2d_1_depthwise_relu6<
8predict_mobilenetv1_mobilenetv1_conv2d_1_pointwise_relu6<
8predict_mobilenetv1_mobilenetv1_conv2d_2_depthwise_relu6<
8predict_mobilenetv1_mobilenetv1_conv2d_2_pointwise_relu6<
8predict_mobilenetv1_mobilenetv1_conv2d_3_depthwise_relu6<
8predict_mobilenetv1_mobilenetv1_conv2d_3_pointwise_relu6<
8predict_mobilenetv1_mobilenetv1_conv2d_4_depthwise_relu6<
8predict_mobilenetv1_mobilenetv1_conv2d_4_pointwise_relu6<
8predict_mobilenetv1_mobilenetv1_conv2d_5_depthwise_relu6<
8predict_mobilenetv1_mobilenetv1_conv2d_5_pointwise_relu6<
8predict_mobilenetv1_mobilenetv1_conv2d_6_depthwise_relu6<
8predict_mobilenetv1_mobilenetv1_conv2d_6_pointwise_relu6<
8predict_mobilenetv1_mobilenetv1_conv2d_7_depthwise_relu6<
8predict_mobilenetv1_mobilenetv1_conv2d_7_pointwise_relu6<
8predict_mobilenetv1_mobilenetv1_conv2d_8_depthwise_relu6<
8predict_mobilenetv1_mobilenetv1_conv2d_8_pointwise_relu6<
8predict_mobilenetv1_mobilenetv1_conv2d_9_depthwise_relu6<
8predict_mobilenetv1_mobilenetv1_conv2d_9_pointwise_relu6/
+predict_mobilenetv1_logits_spatialsqueeze_0-
)predict_mobilenetv1_predictions_reshape_1*
&predict_mobilenetv1_logits_global_pool?w
predict/hub_input/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
predict/hub_input/Mul/y?
predict/hub_input/MulMulplaceholder_1 predict/hub_input/Mul/y:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2
predict/hub_input/Mulw
predict/hub_input/Sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
predict/hub_input/Sub/y?
predict/hub_input/SubSubpredict/hub_input/Mul:z:0 predict/hub_input/Sub/y:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2
predict/hub_input/Sub?
>predict/MobilenetV1/MobilenetV1/Conv2d_0/Conv2D/ReadVariableOpReadVariableOpgtrain_mobilenetv1_mobilenetv1_conv2d_0_kernel_regularizer_l2_regularizer_l2loss_readvariableop_resource*&
_output_shapes
:*
dtype02@
>predict/MobilenetV1/MobilenetV1/Conv2d_0/Conv2D/ReadVariableOp?
/predict/MobilenetV1/MobilenetV1/Conv2d_0/Conv2DConv2Dpredict/hub_input/Sub:z:0Fpredict/MobilenetV1/MobilenetV1/Conv2d_0/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
21
/predict/MobilenetV1/MobilenetV1/Conv2d_0/Conv2D?
Apredict/MobilenetV1/MobilenetV1/Conv2d_0/BatchNorm/ReadVariableOpReadVariableOpHtrain_mobilenetv1_mobilenetv1_conv2d_0_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02C
Apredict/MobilenetV1/MobilenetV1/Conv2d_0/BatchNorm/ReadVariableOp?
Cpredict/MobilenetV1/MobilenetV1/Conv2d_0/BatchNorm/ReadVariableOp_1ReadVariableOpJtrain_mobilenetv1_mobilenetv1_conv2d_0_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02E
Cpredict/MobilenetV1/MobilenetV1/Conv2d_0/BatchNorm/ReadVariableOp_1?
Rpredict/MobilenetV1/MobilenetV1/Conv2d_0/BatchNorm/FusedBatchNormV3/ReadVariableOpReadVariableOpYtrain_mobilenetv1_mobilenetv1_conv2d_0_batchnorm_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02T
Rpredict/MobilenetV1/MobilenetV1/Conv2d_0/BatchNorm/FusedBatchNormV3/ReadVariableOp?
Tpredict/MobilenetV1/MobilenetV1/Conv2d_0/BatchNorm/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp[train_mobilenetv1_mobilenetv1_conv2d_0_batchnorm_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02V
Tpredict/MobilenetV1/MobilenetV1/Conv2d_0/BatchNorm/FusedBatchNormV3/ReadVariableOp_1?
Cpredict/MobilenetV1/MobilenetV1/Conv2d_0/BatchNorm/FusedBatchNormV3FusedBatchNormV38predict/MobilenetV1/MobilenetV1/Conv2d_0/Conv2D:output:0Ipredict/MobilenetV1/MobilenetV1/Conv2d_0/BatchNorm/ReadVariableOp:value:0Kpredict/MobilenetV1/MobilenetV1/Conv2d_0/BatchNorm/ReadVariableOp_1:value:0Zpredict/MobilenetV1/MobilenetV1/Conv2d_0/BatchNorm/FusedBatchNormV3/ReadVariableOp:value:0\predict/MobilenetV1/MobilenetV1/Conv2d_0/BatchNorm/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2E
Cpredict/MobilenetV1/MobilenetV1/Conv2d_0/BatchNorm/FusedBatchNormV3?
.predict/MobilenetV1/MobilenetV1/Conv2d_0/Relu6Relu6Gpredict/MobilenetV1/MobilenetV1/Conv2d_0/BatchNorm/FusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+???????????????????????????20
.predict/MobilenetV1/MobilenetV1/Conv2d_0/Relu6?
Kpredict/MobilenetV1/MobilenetV1/Conv2d_1_depthwise/depthwise/ReadVariableOpReadVariableOpRtrain_mobilenetv1_mobilenetv1_conv2d_1_depthwise_depthwise_readvariableop_resource*&
_output_shapes
:*
dtype02M
Kpredict/MobilenetV1/MobilenetV1/Conv2d_1_depthwise/depthwise/ReadVariableOp?
<predict/MobilenetV1/MobilenetV1/Conv2d_1_depthwise/depthwiseDepthwiseConv2dNative<predict/MobilenetV1/MobilenetV1/Conv2d_0/Relu6:activations:0Spredict/MobilenetV1/MobilenetV1/Conv2d_1_depthwise/depthwise/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
2>
<predict/MobilenetV1/MobilenetV1/Conv2d_1_depthwise/depthwise?
Kpredict/MobilenetV1/MobilenetV1/Conv2d_1_depthwise/BatchNorm/ReadVariableOpReadVariableOpRtrain_mobilenetv1_mobilenetv1_conv2d_1_depthwise_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02M
Kpredict/MobilenetV1/MobilenetV1/Conv2d_1_depthwise/BatchNorm/ReadVariableOp?
Mpredict/MobilenetV1/MobilenetV1/Conv2d_1_depthwise/BatchNorm/ReadVariableOp_1ReadVariableOpTtrain_mobilenetv1_mobilenetv1_conv2d_1_depthwise_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02O
Mpredict/MobilenetV1/MobilenetV1/Conv2d_1_depthwise/BatchNorm/ReadVariableOp_1?
\predict/MobilenetV1/MobilenetV1/Conv2d_1_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOpReadVariableOpctrain_mobilenetv1_mobilenetv1_conv2d_1_depthwise_batchnorm_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02^
\predict/MobilenetV1/MobilenetV1/Conv2d_1_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp?
^predict/MobilenetV1/MobilenetV1/Conv2d_1_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpetrain_mobilenetv1_mobilenetv1_conv2d_1_depthwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02`
^predict/MobilenetV1/MobilenetV1/Conv2d_1_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1?
Mpredict/MobilenetV1/MobilenetV1/Conv2d_1_depthwise/BatchNorm/FusedBatchNormV3FusedBatchNormV3Epredict/MobilenetV1/MobilenetV1/Conv2d_1_depthwise/depthwise:output:0Spredict/MobilenetV1/MobilenetV1/Conv2d_1_depthwise/BatchNorm/ReadVariableOp:value:0Upredict/MobilenetV1/MobilenetV1/Conv2d_1_depthwise/BatchNorm/ReadVariableOp_1:value:0dpredict/MobilenetV1/MobilenetV1/Conv2d_1_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp:value:0fpredict/MobilenetV1/MobilenetV1/Conv2d_1_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2O
Mpredict/MobilenetV1/MobilenetV1/Conv2d_1_depthwise/BatchNorm/FusedBatchNormV3?
8predict/MobilenetV1/MobilenetV1/Conv2d_1_depthwise/Relu6Relu6Qpredict/MobilenetV1/MobilenetV1/Conv2d_1_depthwise/BatchNorm/FusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+???????????????????????????2:
8predict/MobilenetV1/MobilenetV1/Conv2d_1_depthwise/Relu6?
Hpredict/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/Conv2D/ReadVariableOpReadVariableOpqtrain_mobilenetv1_mobilenetv1_conv2d_1_pointwise_kernel_regularizer_l2_regularizer_l2loss_readvariableop_resource*&
_output_shapes
:0*
dtype02J
Hpredict/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/Conv2D/ReadVariableOp?
9predict/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/Conv2DConv2DFpredict/MobilenetV1/MobilenetV1/Conv2d_1_depthwise/Relu6:activations:0Ppredict/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????0*
paddingSAME*
strides
2;
9predict/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/Conv2D?
Kpredict/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/BatchNorm/ReadVariableOpReadVariableOpRtrain_mobilenetv1_mobilenetv1_conv2d_1_pointwise_batchnorm_readvariableop_resource*
_output_shapes
:0*
dtype02M
Kpredict/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/BatchNorm/ReadVariableOp?
Mpredict/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/BatchNorm/ReadVariableOp_1ReadVariableOpTtrain_mobilenetv1_mobilenetv1_conv2d_1_pointwise_batchnorm_readvariableop_1_resource*
_output_shapes
:0*
dtype02O
Mpredict/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/BatchNorm/ReadVariableOp_1?
\predict/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOpReadVariableOpctrain_mobilenetv1_mobilenetv1_conv2d_1_pointwise_batchnorm_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02^
\predict/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp?
^predict/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpetrain_mobilenetv1_mobilenetv1_conv2d_1_pointwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02`
^predict/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1?
Mpredict/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/BatchNorm/FusedBatchNormV3FusedBatchNormV3Bpredict/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/Conv2D:output:0Spredict/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/BatchNorm/ReadVariableOp:value:0Upredict/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/BatchNorm/ReadVariableOp_1:value:0dpredict/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp:value:0fpredict/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????0:0:0:0:0:*
epsilon%o?:*
is_training( 2O
Mpredict/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/BatchNorm/FusedBatchNormV3?
8predict/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/Relu6Relu6Qpredict/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/BatchNorm/FusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+???????????????????????????02:
8predict/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/Relu6?
Kpredict/MobilenetV1/MobilenetV1/Conv2d_2_depthwise/depthwise/ReadVariableOpReadVariableOpRtrain_mobilenetv1_mobilenetv1_conv2d_2_depthwise_depthwise_readvariableop_resource*&
_output_shapes
:0*
dtype02M
Kpredict/MobilenetV1/MobilenetV1/Conv2d_2_depthwise/depthwise/ReadVariableOp?
<predict/MobilenetV1/MobilenetV1/Conv2d_2_depthwise/depthwiseDepthwiseConv2dNativeFpredict/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/Relu6:activations:0Spredict/MobilenetV1/MobilenetV1/Conv2d_2_depthwise/depthwise/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????0*
paddingSAME*
strides
2>
<predict/MobilenetV1/MobilenetV1/Conv2d_2_depthwise/depthwise?
Kpredict/MobilenetV1/MobilenetV1/Conv2d_2_depthwise/BatchNorm/ReadVariableOpReadVariableOpRtrain_mobilenetv1_mobilenetv1_conv2d_2_depthwise_batchnorm_readvariableop_resource*
_output_shapes
:0*
dtype02M
Kpredict/MobilenetV1/MobilenetV1/Conv2d_2_depthwise/BatchNorm/ReadVariableOp?
Mpredict/MobilenetV1/MobilenetV1/Conv2d_2_depthwise/BatchNorm/ReadVariableOp_1ReadVariableOpTtrain_mobilenetv1_mobilenetv1_conv2d_2_depthwise_batchnorm_readvariableop_1_resource*
_output_shapes
:0*
dtype02O
Mpredict/MobilenetV1/MobilenetV1/Conv2d_2_depthwise/BatchNorm/ReadVariableOp_1?
\predict/MobilenetV1/MobilenetV1/Conv2d_2_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOpReadVariableOpctrain_mobilenetv1_mobilenetv1_conv2d_2_depthwise_batchnorm_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02^
\predict/MobilenetV1/MobilenetV1/Conv2d_2_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp?
^predict/MobilenetV1/MobilenetV1/Conv2d_2_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpetrain_mobilenetv1_mobilenetv1_conv2d_2_depthwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02`
^predict/MobilenetV1/MobilenetV1/Conv2d_2_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1?
Mpredict/MobilenetV1/MobilenetV1/Conv2d_2_depthwise/BatchNorm/FusedBatchNormV3FusedBatchNormV3Epredict/MobilenetV1/MobilenetV1/Conv2d_2_depthwise/depthwise:output:0Spredict/MobilenetV1/MobilenetV1/Conv2d_2_depthwise/BatchNorm/ReadVariableOp:value:0Upredict/MobilenetV1/MobilenetV1/Conv2d_2_depthwise/BatchNorm/ReadVariableOp_1:value:0dpredict/MobilenetV1/MobilenetV1/Conv2d_2_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp:value:0fpredict/MobilenetV1/MobilenetV1/Conv2d_2_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????0:0:0:0:0:*
epsilon%o?:*
is_training( 2O
Mpredict/MobilenetV1/MobilenetV1/Conv2d_2_depthwise/BatchNorm/FusedBatchNormV3?
8predict/MobilenetV1/MobilenetV1/Conv2d_2_depthwise/Relu6Relu6Qpredict/MobilenetV1/MobilenetV1/Conv2d_2_depthwise/BatchNorm/FusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+???????????????????????????02:
8predict/MobilenetV1/MobilenetV1/Conv2d_2_depthwise/Relu6?
Hpredict/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/Conv2D/ReadVariableOpReadVariableOpqtrain_mobilenetv1_mobilenetv1_conv2d_2_pointwise_kernel_regularizer_l2_regularizer_l2loss_readvariableop_resource*&
_output_shapes
:0`*
dtype02J
Hpredict/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/Conv2D/ReadVariableOp?
9predict/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/Conv2DConv2DFpredict/MobilenetV1/MobilenetV1/Conv2d_2_depthwise/Relu6:activations:0Ppredict/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????`*
paddingSAME*
strides
2;
9predict/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/Conv2D?
Kpredict/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/BatchNorm/ReadVariableOpReadVariableOpRtrain_mobilenetv1_mobilenetv1_conv2d_2_pointwise_batchnorm_readvariableop_resource*
_output_shapes
:`*
dtype02M
Kpredict/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/BatchNorm/ReadVariableOp?
Mpredict/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/BatchNorm/ReadVariableOp_1ReadVariableOpTtrain_mobilenetv1_mobilenetv1_conv2d_2_pointwise_batchnorm_readvariableop_1_resource*
_output_shapes
:`*
dtype02O
Mpredict/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/BatchNorm/ReadVariableOp_1?
\predict/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOpReadVariableOpctrain_mobilenetv1_mobilenetv1_conv2d_2_pointwise_batchnorm_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:`*
dtype02^
\predict/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp?
^predict/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpetrain_mobilenetv1_mobilenetv1_conv2d_2_pointwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:`*
dtype02`
^predict/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1?
Mpredict/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/BatchNorm/FusedBatchNormV3FusedBatchNormV3Bpredict/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/Conv2D:output:0Spredict/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/BatchNorm/ReadVariableOp:value:0Upredict/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/BatchNorm/ReadVariableOp_1:value:0dpredict/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp:value:0fpredict/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????`:`:`:`:`:*
epsilon%o?:*
is_training( 2O
Mpredict/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/BatchNorm/FusedBatchNormV3?
8predict/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/Relu6Relu6Qpredict/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/BatchNorm/FusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+???????????????????????????`2:
8predict/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/Relu6?
Kpredict/MobilenetV1/MobilenetV1/Conv2d_3_depthwise/depthwise/ReadVariableOpReadVariableOpRtrain_mobilenetv1_mobilenetv1_conv2d_3_depthwise_depthwise_readvariableop_resource*&
_output_shapes
:`*
dtype02M
Kpredict/MobilenetV1/MobilenetV1/Conv2d_3_depthwise/depthwise/ReadVariableOp?
<predict/MobilenetV1/MobilenetV1/Conv2d_3_depthwise/depthwiseDepthwiseConv2dNativeFpredict/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/Relu6:activations:0Spredict/MobilenetV1/MobilenetV1/Conv2d_3_depthwise/depthwise/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????`*
paddingSAME*
strides
2>
<predict/MobilenetV1/MobilenetV1/Conv2d_3_depthwise/depthwise?
Kpredict/MobilenetV1/MobilenetV1/Conv2d_3_depthwise/BatchNorm/ReadVariableOpReadVariableOpRtrain_mobilenetv1_mobilenetv1_conv2d_3_depthwise_batchnorm_readvariableop_resource*
_output_shapes
:`*
dtype02M
Kpredict/MobilenetV1/MobilenetV1/Conv2d_3_depthwise/BatchNorm/ReadVariableOp?
Mpredict/MobilenetV1/MobilenetV1/Conv2d_3_depthwise/BatchNorm/ReadVariableOp_1ReadVariableOpTtrain_mobilenetv1_mobilenetv1_conv2d_3_depthwise_batchnorm_readvariableop_1_resource*
_output_shapes
:`*
dtype02O
Mpredict/MobilenetV1/MobilenetV1/Conv2d_3_depthwise/BatchNorm/ReadVariableOp_1?
\predict/MobilenetV1/MobilenetV1/Conv2d_3_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOpReadVariableOpctrain_mobilenetv1_mobilenetv1_conv2d_3_depthwise_batchnorm_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:`*
dtype02^
\predict/MobilenetV1/MobilenetV1/Conv2d_3_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp?
^predict/MobilenetV1/MobilenetV1/Conv2d_3_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpetrain_mobilenetv1_mobilenetv1_conv2d_3_depthwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:`*
dtype02`
^predict/MobilenetV1/MobilenetV1/Conv2d_3_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1?
Mpredict/MobilenetV1/MobilenetV1/Conv2d_3_depthwise/BatchNorm/FusedBatchNormV3FusedBatchNormV3Epredict/MobilenetV1/MobilenetV1/Conv2d_3_depthwise/depthwise:output:0Spredict/MobilenetV1/MobilenetV1/Conv2d_3_depthwise/BatchNorm/ReadVariableOp:value:0Upredict/MobilenetV1/MobilenetV1/Conv2d_3_depthwise/BatchNorm/ReadVariableOp_1:value:0dpredict/MobilenetV1/MobilenetV1/Conv2d_3_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp:value:0fpredict/MobilenetV1/MobilenetV1/Conv2d_3_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????`:`:`:`:`:*
epsilon%o?:*
is_training( 2O
Mpredict/MobilenetV1/MobilenetV1/Conv2d_3_depthwise/BatchNorm/FusedBatchNormV3?
8predict/MobilenetV1/MobilenetV1/Conv2d_3_depthwise/Relu6Relu6Qpredict/MobilenetV1/MobilenetV1/Conv2d_3_depthwise/BatchNorm/FusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+???????????????????????????`2:
8predict/MobilenetV1/MobilenetV1/Conv2d_3_depthwise/Relu6?
Hpredict/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/Conv2D/ReadVariableOpReadVariableOpqtrain_mobilenetv1_mobilenetv1_conv2d_3_pointwise_kernel_regularizer_l2_regularizer_l2loss_readvariableop_resource*&
_output_shapes
:``*
dtype02J
Hpredict/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/Conv2D/ReadVariableOp?
9predict/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/Conv2DConv2DFpredict/MobilenetV1/MobilenetV1/Conv2d_3_depthwise/Relu6:activations:0Ppredict/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????`*
paddingSAME*
strides
2;
9predict/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/Conv2D?
Kpredict/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/BatchNorm/ReadVariableOpReadVariableOpRtrain_mobilenetv1_mobilenetv1_conv2d_3_pointwise_batchnorm_readvariableop_resource*
_output_shapes
:`*
dtype02M
Kpredict/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/BatchNorm/ReadVariableOp?
Mpredict/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/BatchNorm/ReadVariableOp_1ReadVariableOpTtrain_mobilenetv1_mobilenetv1_conv2d_3_pointwise_batchnorm_readvariableop_1_resource*
_output_shapes
:`*
dtype02O
Mpredict/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/BatchNorm/ReadVariableOp_1?
\predict/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOpReadVariableOpctrain_mobilenetv1_mobilenetv1_conv2d_3_pointwise_batchnorm_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:`*
dtype02^
\predict/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp?
^predict/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpetrain_mobilenetv1_mobilenetv1_conv2d_3_pointwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:`*
dtype02`
^predict/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1?
Mpredict/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/BatchNorm/FusedBatchNormV3FusedBatchNormV3Bpredict/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/Conv2D:output:0Spredict/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/BatchNorm/ReadVariableOp:value:0Upredict/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/BatchNorm/ReadVariableOp_1:value:0dpredict/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp:value:0fpredict/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????`:`:`:`:`:*
epsilon%o?:*
is_training( 2O
Mpredict/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/BatchNorm/FusedBatchNormV3?
8predict/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/Relu6Relu6Qpredict/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/BatchNorm/FusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+???????????????????????????`2:
8predict/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/Relu6?
Kpredict/MobilenetV1/MobilenetV1/Conv2d_4_depthwise/depthwise/ReadVariableOpReadVariableOpRtrain_mobilenetv1_mobilenetv1_conv2d_4_depthwise_depthwise_readvariableop_resource*&
_output_shapes
:`*
dtype02M
Kpredict/MobilenetV1/MobilenetV1/Conv2d_4_depthwise/depthwise/ReadVariableOp?
<predict/MobilenetV1/MobilenetV1/Conv2d_4_depthwise/depthwiseDepthwiseConv2dNativeFpredict/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/Relu6:activations:0Spredict/MobilenetV1/MobilenetV1/Conv2d_4_depthwise/depthwise/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????`*
paddingSAME*
strides
2>
<predict/MobilenetV1/MobilenetV1/Conv2d_4_depthwise/depthwise?
Kpredict/MobilenetV1/MobilenetV1/Conv2d_4_depthwise/BatchNorm/ReadVariableOpReadVariableOpRtrain_mobilenetv1_mobilenetv1_conv2d_4_depthwise_batchnorm_readvariableop_resource*
_output_shapes
:`*
dtype02M
Kpredict/MobilenetV1/MobilenetV1/Conv2d_4_depthwise/BatchNorm/ReadVariableOp?
Mpredict/MobilenetV1/MobilenetV1/Conv2d_4_depthwise/BatchNorm/ReadVariableOp_1ReadVariableOpTtrain_mobilenetv1_mobilenetv1_conv2d_4_depthwise_batchnorm_readvariableop_1_resource*
_output_shapes
:`*
dtype02O
Mpredict/MobilenetV1/MobilenetV1/Conv2d_4_depthwise/BatchNorm/ReadVariableOp_1?
\predict/MobilenetV1/MobilenetV1/Conv2d_4_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOpReadVariableOpctrain_mobilenetv1_mobilenetv1_conv2d_4_depthwise_batchnorm_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:`*
dtype02^
\predict/MobilenetV1/MobilenetV1/Conv2d_4_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp?
^predict/MobilenetV1/MobilenetV1/Conv2d_4_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpetrain_mobilenetv1_mobilenetv1_conv2d_4_depthwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:`*
dtype02`
^predict/MobilenetV1/MobilenetV1/Conv2d_4_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1?
Mpredict/MobilenetV1/MobilenetV1/Conv2d_4_depthwise/BatchNorm/FusedBatchNormV3FusedBatchNormV3Epredict/MobilenetV1/MobilenetV1/Conv2d_4_depthwise/depthwise:output:0Spredict/MobilenetV1/MobilenetV1/Conv2d_4_depthwise/BatchNorm/ReadVariableOp:value:0Upredict/MobilenetV1/MobilenetV1/Conv2d_4_depthwise/BatchNorm/ReadVariableOp_1:value:0dpredict/MobilenetV1/MobilenetV1/Conv2d_4_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp:value:0fpredict/MobilenetV1/MobilenetV1/Conv2d_4_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????`:`:`:`:`:*
epsilon%o?:*
is_training( 2O
Mpredict/MobilenetV1/MobilenetV1/Conv2d_4_depthwise/BatchNorm/FusedBatchNormV3?
8predict/MobilenetV1/MobilenetV1/Conv2d_4_depthwise/Relu6Relu6Qpredict/MobilenetV1/MobilenetV1/Conv2d_4_depthwise/BatchNorm/FusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+???????????????????????????`2:
8predict/MobilenetV1/MobilenetV1/Conv2d_4_depthwise/Relu6?
Hpredict/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/Conv2D/ReadVariableOpReadVariableOpqtrain_mobilenetv1_mobilenetv1_conv2d_4_pointwise_kernel_regularizer_l2_regularizer_l2loss_readvariableop_resource*'
_output_shapes
:`?*
dtype02J
Hpredict/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/Conv2D/ReadVariableOp?
9predict/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/Conv2DConv2DFpredict/MobilenetV1/MobilenetV1/Conv2d_4_depthwise/Relu6:activations:0Ppredict/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
2;
9predict/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/Conv2D?
Kpredict/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/BatchNorm/ReadVariableOpReadVariableOpRtrain_mobilenetv1_mobilenetv1_conv2d_4_pointwise_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02M
Kpredict/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/BatchNorm/ReadVariableOp?
Mpredict/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/BatchNorm/ReadVariableOp_1ReadVariableOpTtrain_mobilenetv1_mobilenetv1_conv2d_4_pointwise_batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype02O
Mpredict/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/BatchNorm/ReadVariableOp_1?
\predict/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOpReadVariableOpctrain_mobilenetv1_mobilenetv1_conv2d_4_pointwise_batchnorm_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02^
\predict/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp?
^predict/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpetrain_mobilenetv1_mobilenetv1_conv2d_4_pointwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02`
^predict/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1?
Mpredict/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/BatchNorm/FusedBatchNormV3FusedBatchNormV3Bpredict/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/Conv2D:output:0Spredict/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/BatchNorm/ReadVariableOp:value:0Upredict/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/BatchNorm/ReadVariableOp_1:value:0dpredict/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp:value:0fpredict/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( 2O
Mpredict/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/BatchNorm/FusedBatchNormV3?
8predict/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/Relu6Relu6Qpredict/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/BatchNorm/FusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2:
8predict/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/Relu6?
Kpredict/MobilenetV1/MobilenetV1/Conv2d_5_depthwise/depthwise/ReadVariableOpReadVariableOpRtrain_mobilenetv1_mobilenetv1_conv2d_5_depthwise_depthwise_readvariableop_resource*'
_output_shapes
:?*
dtype02M
Kpredict/MobilenetV1/MobilenetV1/Conv2d_5_depthwise/depthwise/ReadVariableOp?
<predict/MobilenetV1/MobilenetV1/Conv2d_5_depthwise/depthwiseDepthwiseConv2dNativeFpredict/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/Relu6:activations:0Spredict/MobilenetV1/MobilenetV1/Conv2d_5_depthwise/depthwise/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
2>
<predict/MobilenetV1/MobilenetV1/Conv2d_5_depthwise/depthwise?
Kpredict/MobilenetV1/MobilenetV1/Conv2d_5_depthwise/BatchNorm/ReadVariableOpReadVariableOpRtrain_mobilenetv1_mobilenetv1_conv2d_5_depthwise_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02M
Kpredict/MobilenetV1/MobilenetV1/Conv2d_5_depthwise/BatchNorm/ReadVariableOp?
Mpredict/MobilenetV1/MobilenetV1/Conv2d_5_depthwise/BatchNorm/ReadVariableOp_1ReadVariableOpTtrain_mobilenetv1_mobilenetv1_conv2d_5_depthwise_batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype02O
Mpredict/MobilenetV1/MobilenetV1/Conv2d_5_depthwise/BatchNorm/ReadVariableOp_1?
\predict/MobilenetV1/MobilenetV1/Conv2d_5_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOpReadVariableOpctrain_mobilenetv1_mobilenetv1_conv2d_5_depthwise_batchnorm_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02^
\predict/MobilenetV1/MobilenetV1/Conv2d_5_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp?
^predict/MobilenetV1/MobilenetV1/Conv2d_5_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpetrain_mobilenetv1_mobilenetv1_conv2d_5_depthwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02`
^predict/MobilenetV1/MobilenetV1/Conv2d_5_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1?
Mpredict/MobilenetV1/MobilenetV1/Conv2d_5_depthwise/BatchNorm/FusedBatchNormV3FusedBatchNormV3Epredict/MobilenetV1/MobilenetV1/Conv2d_5_depthwise/depthwise:output:0Spredict/MobilenetV1/MobilenetV1/Conv2d_5_depthwise/BatchNorm/ReadVariableOp:value:0Upredict/MobilenetV1/MobilenetV1/Conv2d_5_depthwise/BatchNorm/ReadVariableOp_1:value:0dpredict/MobilenetV1/MobilenetV1/Conv2d_5_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp:value:0fpredict/MobilenetV1/MobilenetV1/Conv2d_5_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( 2O
Mpredict/MobilenetV1/MobilenetV1/Conv2d_5_depthwise/BatchNorm/FusedBatchNormV3?
8predict/MobilenetV1/MobilenetV1/Conv2d_5_depthwise/Relu6Relu6Qpredict/MobilenetV1/MobilenetV1/Conv2d_5_depthwise/BatchNorm/FusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2:
8predict/MobilenetV1/MobilenetV1/Conv2d_5_depthwise/Relu6?
Hpredict/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/Conv2D/ReadVariableOpReadVariableOpqtrain_mobilenetv1_mobilenetv1_conv2d_5_pointwise_kernel_regularizer_l2_regularizer_l2loss_readvariableop_resource*(
_output_shapes
:??*
dtype02J
Hpredict/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/Conv2D/ReadVariableOp?
9predict/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/Conv2DConv2DFpredict/MobilenetV1/MobilenetV1/Conv2d_5_depthwise/Relu6:activations:0Ppredict/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
2;
9predict/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/Conv2D?
Kpredict/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/BatchNorm/ReadVariableOpReadVariableOpRtrain_mobilenetv1_mobilenetv1_conv2d_5_pointwise_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02M
Kpredict/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/BatchNorm/ReadVariableOp?
Mpredict/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/BatchNorm/ReadVariableOp_1ReadVariableOpTtrain_mobilenetv1_mobilenetv1_conv2d_5_pointwise_batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype02O
Mpredict/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/BatchNorm/ReadVariableOp_1?
\predict/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOpReadVariableOpctrain_mobilenetv1_mobilenetv1_conv2d_5_pointwise_batchnorm_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02^
\predict/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp?
^predict/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpetrain_mobilenetv1_mobilenetv1_conv2d_5_pointwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02`
^predict/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1?
Mpredict/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/BatchNorm/FusedBatchNormV3FusedBatchNormV3Bpredict/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/Conv2D:output:0Spredict/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/BatchNorm/ReadVariableOp:value:0Upredict/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/BatchNorm/ReadVariableOp_1:value:0dpredict/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp:value:0fpredict/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( 2O
Mpredict/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/BatchNorm/FusedBatchNormV3?
8predict/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/Relu6Relu6Qpredict/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/BatchNorm/FusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2:
8predict/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/Relu6?
Kpredict/MobilenetV1/MobilenetV1/Conv2d_6_depthwise/depthwise/ReadVariableOpReadVariableOpRtrain_mobilenetv1_mobilenetv1_conv2d_6_depthwise_depthwise_readvariableop_resource*'
_output_shapes
:?*
dtype02M
Kpredict/MobilenetV1/MobilenetV1/Conv2d_6_depthwise/depthwise/ReadVariableOp?
<predict/MobilenetV1/MobilenetV1/Conv2d_6_depthwise/depthwiseDepthwiseConv2dNativeFpredict/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/Relu6:activations:0Spredict/MobilenetV1/MobilenetV1/Conv2d_6_depthwise/depthwise/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
2>
<predict/MobilenetV1/MobilenetV1/Conv2d_6_depthwise/depthwise?
Kpredict/MobilenetV1/MobilenetV1/Conv2d_6_depthwise/BatchNorm/ReadVariableOpReadVariableOpRtrain_mobilenetv1_mobilenetv1_conv2d_6_depthwise_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02M
Kpredict/MobilenetV1/MobilenetV1/Conv2d_6_depthwise/BatchNorm/ReadVariableOp?
Mpredict/MobilenetV1/MobilenetV1/Conv2d_6_depthwise/BatchNorm/ReadVariableOp_1ReadVariableOpTtrain_mobilenetv1_mobilenetv1_conv2d_6_depthwise_batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype02O
Mpredict/MobilenetV1/MobilenetV1/Conv2d_6_depthwise/BatchNorm/ReadVariableOp_1?
\predict/MobilenetV1/MobilenetV1/Conv2d_6_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOpReadVariableOpctrain_mobilenetv1_mobilenetv1_conv2d_6_depthwise_batchnorm_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02^
\predict/MobilenetV1/MobilenetV1/Conv2d_6_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp?
^predict/MobilenetV1/MobilenetV1/Conv2d_6_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpetrain_mobilenetv1_mobilenetv1_conv2d_6_depthwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02`
^predict/MobilenetV1/MobilenetV1/Conv2d_6_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1?
Mpredict/MobilenetV1/MobilenetV1/Conv2d_6_depthwise/BatchNorm/FusedBatchNormV3FusedBatchNormV3Epredict/MobilenetV1/MobilenetV1/Conv2d_6_depthwise/depthwise:output:0Spredict/MobilenetV1/MobilenetV1/Conv2d_6_depthwise/BatchNorm/ReadVariableOp:value:0Upredict/MobilenetV1/MobilenetV1/Conv2d_6_depthwise/BatchNorm/ReadVariableOp_1:value:0dpredict/MobilenetV1/MobilenetV1/Conv2d_6_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp:value:0fpredict/MobilenetV1/MobilenetV1/Conv2d_6_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( 2O
Mpredict/MobilenetV1/MobilenetV1/Conv2d_6_depthwise/BatchNorm/FusedBatchNormV3?
8predict/MobilenetV1/MobilenetV1/Conv2d_6_depthwise/Relu6Relu6Qpredict/MobilenetV1/MobilenetV1/Conv2d_6_depthwise/BatchNorm/FusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2:
8predict/MobilenetV1/MobilenetV1/Conv2d_6_depthwise/Relu6?
Hpredict/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/Conv2D/ReadVariableOpReadVariableOpqtrain_mobilenetv1_mobilenetv1_conv2d_6_pointwise_kernel_regularizer_l2_regularizer_l2loss_readvariableop_resource*(
_output_shapes
:??*
dtype02J
Hpredict/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/Conv2D/ReadVariableOp?
9predict/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/Conv2DConv2DFpredict/MobilenetV1/MobilenetV1/Conv2d_6_depthwise/Relu6:activations:0Ppredict/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
2;
9predict/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/Conv2D?
Kpredict/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/BatchNorm/ReadVariableOpReadVariableOpRtrain_mobilenetv1_mobilenetv1_conv2d_6_pointwise_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02M
Kpredict/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/BatchNorm/ReadVariableOp?
Mpredict/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/BatchNorm/ReadVariableOp_1ReadVariableOpTtrain_mobilenetv1_mobilenetv1_conv2d_6_pointwise_batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype02O
Mpredict/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/BatchNorm/ReadVariableOp_1?
\predict/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOpReadVariableOpctrain_mobilenetv1_mobilenetv1_conv2d_6_pointwise_batchnorm_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02^
\predict/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp?
^predict/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpetrain_mobilenetv1_mobilenetv1_conv2d_6_pointwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02`
^predict/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1?
Mpredict/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/BatchNorm/FusedBatchNormV3FusedBatchNormV3Bpredict/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/Conv2D:output:0Spredict/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/BatchNorm/ReadVariableOp:value:0Upredict/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/BatchNorm/ReadVariableOp_1:value:0dpredict/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp:value:0fpredict/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( 2O
Mpredict/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/BatchNorm/FusedBatchNormV3?
8predict/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/Relu6Relu6Qpredict/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/BatchNorm/FusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2:
8predict/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/Relu6?
Kpredict/MobilenetV1/MobilenetV1/Conv2d_7_depthwise/depthwise/ReadVariableOpReadVariableOpRtrain_mobilenetv1_mobilenetv1_conv2d_7_depthwise_depthwise_readvariableop_resource*'
_output_shapes
:?*
dtype02M
Kpredict/MobilenetV1/MobilenetV1/Conv2d_7_depthwise/depthwise/ReadVariableOp?
<predict/MobilenetV1/MobilenetV1/Conv2d_7_depthwise/depthwiseDepthwiseConv2dNativeFpredict/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/Relu6:activations:0Spredict/MobilenetV1/MobilenetV1/Conv2d_7_depthwise/depthwise/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
2>
<predict/MobilenetV1/MobilenetV1/Conv2d_7_depthwise/depthwise?
Kpredict/MobilenetV1/MobilenetV1/Conv2d_7_depthwise/BatchNorm/ReadVariableOpReadVariableOpRtrain_mobilenetv1_mobilenetv1_conv2d_7_depthwise_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02M
Kpredict/MobilenetV1/MobilenetV1/Conv2d_7_depthwise/BatchNorm/ReadVariableOp?
Mpredict/MobilenetV1/MobilenetV1/Conv2d_7_depthwise/BatchNorm/ReadVariableOp_1ReadVariableOpTtrain_mobilenetv1_mobilenetv1_conv2d_7_depthwise_batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype02O
Mpredict/MobilenetV1/MobilenetV1/Conv2d_7_depthwise/BatchNorm/ReadVariableOp_1?
\predict/MobilenetV1/MobilenetV1/Conv2d_7_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOpReadVariableOpctrain_mobilenetv1_mobilenetv1_conv2d_7_depthwise_batchnorm_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02^
\predict/MobilenetV1/MobilenetV1/Conv2d_7_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp?
^predict/MobilenetV1/MobilenetV1/Conv2d_7_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpetrain_mobilenetv1_mobilenetv1_conv2d_7_depthwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02`
^predict/MobilenetV1/MobilenetV1/Conv2d_7_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1?
Mpredict/MobilenetV1/MobilenetV1/Conv2d_7_depthwise/BatchNorm/FusedBatchNormV3FusedBatchNormV3Epredict/MobilenetV1/MobilenetV1/Conv2d_7_depthwise/depthwise:output:0Spredict/MobilenetV1/MobilenetV1/Conv2d_7_depthwise/BatchNorm/ReadVariableOp:value:0Upredict/MobilenetV1/MobilenetV1/Conv2d_7_depthwise/BatchNorm/ReadVariableOp_1:value:0dpredict/MobilenetV1/MobilenetV1/Conv2d_7_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp:value:0fpredict/MobilenetV1/MobilenetV1/Conv2d_7_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( 2O
Mpredict/MobilenetV1/MobilenetV1/Conv2d_7_depthwise/BatchNorm/FusedBatchNormV3?
8predict/MobilenetV1/MobilenetV1/Conv2d_7_depthwise/Relu6Relu6Qpredict/MobilenetV1/MobilenetV1/Conv2d_7_depthwise/BatchNorm/FusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2:
8predict/MobilenetV1/MobilenetV1/Conv2d_7_depthwise/Relu6?
Hpredict/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/Conv2D/ReadVariableOpReadVariableOpqtrain_mobilenetv1_mobilenetv1_conv2d_7_pointwise_kernel_regularizer_l2_regularizer_l2loss_readvariableop_resource*(
_output_shapes
:??*
dtype02J
Hpredict/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/Conv2D/ReadVariableOp?
9predict/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/Conv2DConv2DFpredict/MobilenetV1/MobilenetV1/Conv2d_7_depthwise/Relu6:activations:0Ppredict/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
2;
9predict/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/Conv2D?
Kpredict/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/BatchNorm/ReadVariableOpReadVariableOpRtrain_mobilenetv1_mobilenetv1_conv2d_7_pointwise_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02M
Kpredict/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/BatchNorm/ReadVariableOp?
Mpredict/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/BatchNorm/ReadVariableOp_1ReadVariableOpTtrain_mobilenetv1_mobilenetv1_conv2d_7_pointwise_batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype02O
Mpredict/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/BatchNorm/ReadVariableOp_1?
\predict/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOpReadVariableOpctrain_mobilenetv1_mobilenetv1_conv2d_7_pointwise_batchnorm_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02^
\predict/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp?
^predict/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpetrain_mobilenetv1_mobilenetv1_conv2d_7_pointwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02`
^predict/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1?
Mpredict/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/BatchNorm/FusedBatchNormV3FusedBatchNormV3Bpredict/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/Conv2D:output:0Spredict/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/BatchNorm/ReadVariableOp:value:0Upredict/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/BatchNorm/ReadVariableOp_1:value:0dpredict/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp:value:0fpredict/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( 2O
Mpredict/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/BatchNorm/FusedBatchNormV3?
8predict/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/Relu6Relu6Qpredict/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/BatchNorm/FusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2:
8predict/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/Relu6?
Kpredict/MobilenetV1/MobilenetV1/Conv2d_8_depthwise/depthwise/ReadVariableOpReadVariableOpRtrain_mobilenetv1_mobilenetv1_conv2d_8_depthwise_depthwise_readvariableop_resource*'
_output_shapes
:?*
dtype02M
Kpredict/MobilenetV1/MobilenetV1/Conv2d_8_depthwise/depthwise/ReadVariableOp?
<predict/MobilenetV1/MobilenetV1/Conv2d_8_depthwise/depthwiseDepthwiseConv2dNativeFpredict/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/Relu6:activations:0Spredict/MobilenetV1/MobilenetV1/Conv2d_8_depthwise/depthwise/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
2>
<predict/MobilenetV1/MobilenetV1/Conv2d_8_depthwise/depthwise?
Kpredict/MobilenetV1/MobilenetV1/Conv2d_8_depthwise/BatchNorm/ReadVariableOpReadVariableOpRtrain_mobilenetv1_mobilenetv1_conv2d_8_depthwise_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02M
Kpredict/MobilenetV1/MobilenetV1/Conv2d_8_depthwise/BatchNorm/ReadVariableOp?
Mpredict/MobilenetV1/MobilenetV1/Conv2d_8_depthwise/BatchNorm/ReadVariableOp_1ReadVariableOpTtrain_mobilenetv1_mobilenetv1_conv2d_8_depthwise_batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype02O
Mpredict/MobilenetV1/MobilenetV1/Conv2d_8_depthwise/BatchNorm/ReadVariableOp_1?
\predict/MobilenetV1/MobilenetV1/Conv2d_8_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOpReadVariableOpctrain_mobilenetv1_mobilenetv1_conv2d_8_depthwise_batchnorm_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02^
\predict/MobilenetV1/MobilenetV1/Conv2d_8_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp?
^predict/MobilenetV1/MobilenetV1/Conv2d_8_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpetrain_mobilenetv1_mobilenetv1_conv2d_8_depthwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02`
^predict/MobilenetV1/MobilenetV1/Conv2d_8_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1?
Mpredict/MobilenetV1/MobilenetV1/Conv2d_8_depthwise/BatchNorm/FusedBatchNormV3FusedBatchNormV3Epredict/MobilenetV1/MobilenetV1/Conv2d_8_depthwise/depthwise:output:0Spredict/MobilenetV1/MobilenetV1/Conv2d_8_depthwise/BatchNorm/ReadVariableOp:value:0Upredict/MobilenetV1/MobilenetV1/Conv2d_8_depthwise/BatchNorm/ReadVariableOp_1:value:0dpredict/MobilenetV1/MobilenetV1/Conv2d_8_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp:value:0fpredict/MobilenetV1/MobilenetV1/Conv2d_8_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( 2O
Mpredict/MobilenetV1/MobilenetV1/Conv2d_8_depthwise/BatchNorm/FusedBatchNormV3?
8predict/MobilenetV1/MobilenetV1/Conv2d_8_depthwise/Relu6Relu6Qpredict/MobilenetV1/MobilenetV1/Conv2d_8_depthwise/BatchNorm/FusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2:
8predict/MobilenetV1/MobilenetV1/Conv2d_8_depthwise/Relu6?
Hpredict/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/Conv2D/ReadVariableOpReadVariableOpqtrain_mobilenetv1_mobilenetv1_conv2d_8_pointwise_kernel_regularizer_l2_regularizer_l2loss_readvariableop_resource*(
_output_shapes
:??*
dtype02J
Hpredict/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/Conv2D/ReadVariableOp?
9predict/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/Conv2DConv2DFpredict/MobilenetV1/MobilenetV1/Conv2d_8_depthwise/Relu6:activations:0Ppredict/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
2;
9predict/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/Conv2D?
Kpredict/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/BatchNorm/ReadVariableOpReadVariableOpRtrain_mobilenetv1_mobilenetv1_conv2d_8_pointwise_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02M
Kpredict/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/BatchNorm/ReadVariableOp?
Mpredict/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/BatchNorm/ReadVariableOp_1ReadVariableOpTtrain_mobilenetv1_mobilenetv1_conv2d_8_pointwise_batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype02O
Mpredict/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/BatchNorm/ReadVariableOp_1?
\predict/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOpReadVariableOpctrain_mobilenetv1_mobilenetv1_conv2d_8_pointwise_batchnorm_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02^
\predict/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp?
^predict/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpetrain_mobilenetv1_mobilenetv1_conv2d_8_pointwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02`
^predict/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1?
Mpredict/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/BatchNorm/FusedBatchNormV3FusedBatchNormV3Bpredict/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/Conv2D:output:0Spredict/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/BatchNorm/ReadVariableOp:value:0Upredict/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/BatchNorm/ReadVariableOp_1:value:0dpredict/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp:value:0fpredict/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( 2O
Mpredict/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/BatchNorm/FusedBatchNormV3?
8predict/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/Relu6Relu6Qpredict/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/BatchNorm/FusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2:
8predict/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/Relu6?
Kpredict/MobilenetV1/MobilenetV1/Conv2d_9_depthwise/depthwise/ReadVariableOpReadVariableOpRtrain_mobilenetv1_mobilenetv1_conv2d_9_depthwise_depthwise_readvariableop_resource*'
_output_shapes
:?*
dtype02M
Kpredict/MobilenetV1/MobilenetV1/Conv2d_9_depthwise/depthwise/ReadVariableOp?
<predict/MobilenetV1/MobilenetV1/Conv2d_9_depthwise/depthwiseDepthwiseConv2dNativeFpredict/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/Relu6:activations:0Spredict/MobilenetV1/MobilenetV1/Conv2d_9_depthwise/depthwise/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
2>
<predict/MobilenetV1/MobilenetV1/Conv2d_9_depthwise/depthwise?
Kpredict/MobilenetV1/MobilenetV1/Conv2d_9_depthwise/BatchNorm/ReadVariableOpReadVariableOpRtrain_mobilenetv1_mobilenetv1_conv2d_9_depthwise_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02M
Kpredict/MobilenetV1/MobilenetV1/Conv2d_9_depthwise/BatchNorm/ReadVariableOp?
Mpredict/MobilenetV1/MobilenetV1/Conv2d_9_depthwise/BatchNorm/ReadVariableOp_1ReadVariableOpTtrain_mobilenetv1_mobilenetv1_conv2d_9_depthwise_batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype02O
Mpredict/MobilenetV1/MobilenetV1/Conv2d_9_depthwise/BatchNorm/ReadVariableOp_1?
\predict/MobilenetV1/MobilenetV1/Conv2d_9_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOpReadVariableOpctrain_mobilenetv1_mobilenetv1_conv2d_9_depthwise_batchnorm_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02^
\predict/MobilenetV1/MobilenetV1/Conv2d_9_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp?
^predict/MobilenetV1/MobilenetV1/Conv2d_9_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpetrain_mobilenetv1_mobilenetv1_conv2d_9_depthwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02`
^predict/MobilenetV1/MobilenetV1/Conv2d_9_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1?
Mpredict/MobilenetV1/MobilenetV1/Conv2d_9_depthwise/BatchNorm/FusedBatchNormV3FusedBatchNormV3Epredict/MobilenetV1/MobilenetV1/Conv2d_9_depthwise/depthwise:output:0Spredict/MobilenetV1/MobilenetV1/Conv2d_9_depthwise/BatchNorm/ReadVariableOp:value:0Upredict/MobilenetV1/MobilenetV1/Conv2d_9_depthwise/BatchNorm/ReadVariableOp_1:value:0dpredict/MobilenetV1/MobilenetV1/Conv2d_9_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp:value:0fpredict/MobilenetV1/MobilenetV1/Conv2d_9_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( 2O
Mpredict/MobilenetV1/MobilenetV1/Conv2d_9_depthwise/BatchNorm/FusedBatchNormV3?
8predict/MobilenetV1/MobilenetV1/Conv2d_9_depthwise/Relu6Relu6Qpredict/MobilenetV1/MobilenetV1/Conv2d_9_depthwise/BatchNorm/FusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2:
8predict/MobilenetV1/MobilenetV1/Conv2d_9_depthwise/Relu6?
Hpredict/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/Conv2D/ReadVariableOpReadVariableOpqtrain_mobilenetv1_mobilenetv1_conv2d_9_pointwise_kernel_regularizer_l2_regularizer_l2loss_readvariableop_resource*(
_output_shapes
:??*
dtype02J
Hpredict/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/Conv2D/ReadVariableOp?
9predict/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/Conv2DConv2DFpredict/MobilenetV1/MobilenetV1/Conv2d_9_depthwise/Relu6:activations:0Ppredict/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
2;
9predict/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/Conv2D?
Kpredict/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/BatchNorm/ReadVariableOpReadVariableOpRtrain_mobilenetv1_mobilenetv1_conv2d_9_pointwise_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02M
Kpredict/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/BatchNorm/ReadVariableOp?
Mpredict/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/BatchNorm/ReadVariableOp_1ReadVariableOpTtrain_mobilenetv1_mobilenetv1_conv2d_9_pointwise_batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype02O
Mpredict/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/BatchNorm/ReadVariableOp_1?
\predict/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOpReadVariableOpctrain_mobilenetv1_mobilenetv1_conv2d_9_pointwise_batchnorm_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02^
\predict/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp?
^predict/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpetrain_mobilenetv1_mobilenetv1_conv2d_9_pointwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02`
^predict/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1?
Mpredict/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/BatchNorm/FusedBatchNormV3FusedBatchNormV3Bpredict/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/Conv2D:output:0Spredict/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/BatchNorm/ReadVariableOp:value:0Upredict/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/BatchNorm/ReadVariableOp_1:value:0dpredict/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp:value:0fpredict/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( 2O
Mpredict/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/BatchNorm/FusedBatchNormV3?
8predict/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/Relu6Relu6Qpredict/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/BatchNorm/FusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2:
8predict/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/Relu6?
Lpredict/MobilenetV1/MobilenetV1/Conv2d_10_depthwise/depthwise/ReadVariableOpReadVariableOpStrain_mobilenetv1_mobilenetv1_conv2d_10_depthwise_depthwise_readvariableop_resource*'
_output_shapes
:?*
dtype02N
Lpredict/MobilenetV1/MobilenetV1/Conv2d_10_depthwise/depthwise/ReadVariableOp?
=predict/MobilenetV1/MobilenetV1/Conv2d_10_depthwise/depthwiseDepthwiseConv2dNativeFpredict/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/Relu6:activations:0Tpredict/MobilenetV1/MobilenetV1/Conv2d_10_depthwise/depthwise/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
2?
=predict/MobilenetV1/MobilenetV1/Conv2d_10_depthwise/depthwise?
Lpredict/MobilenetV1/MobilenetV1/Conv2d_10_depthwise/BatchNorm/ReadVariableOpReadVariableOpStrain_mobilenetv1_mobilenetv1_conv2d_10_depthwise_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02N
Lpredict/MobilenetV1/MobilenetV1/Conv2d_10_depthwise/BatchNorm/ReadVariableOp?
Npredict/MobilenetV1/MobilenetV1/Conv2d_10_depthwise/BatchNorm/ReadVariableOp_1ReadVariableOpUtrain_mobilenetv1_mobilenetv1_conv2d_10_depthwise_batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype02P
Npredict/MobilenetV1/MobilenetV1/Conv2d_10_depthwise/BatchNorm/ReadVariableOp_1?
]predict/MobilenetV1/MobilenetV1/Conv2d_10_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOpReadVariableOpdtrain_mobilenetv1_mobilenetv1_conv2d_10_depthwise_batchnorm_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02_
]predict/MobilenetV1/MobilenetV1/Conv2d_10_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp?
_predict/MobilenetV1/MobilenetV1/Conv2d_10_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpftrain_mobilenetv1_mobilenetv1_conv2d_10_depthwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02a
_predict/MobilenetV1/MobilenetV1/Conv2d_10_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1?
Npredict/MobilenetV1/MobilenetV1/Conv2d_10_depthwise/BatchNorm/FusedBatchNormV3FusedBatchNormV3Fpredict/MobilenetV1/MobilenetV1/Conv2d_10_depthwise/depthwise:output:0Tpredict/MobilenetV1/MobilenetV1/Conv2d_10_depthwise/BatchNorm/ReadVariableOp:value:0Vpredict/MobilenetV1/MobilenetV1/Conv2d_10_depthwise/BatchNorm/ReadVariableOp_1:value:0epredict/MobilenetV1/MobilenetV1/Conv2d_10_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp:value:0gpredict/MobilenetV1/MobilenetV1/Conv2d_10_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( 2P
Npredict/MobilenetV1/MobilenetV1/Conv2d_10_depthwise/BatchNorm/FusedBatchNormV3?
9predict/MobilenetV1/MobilenetV1/Conv2d_10_depthwise/Relu6Relu6Rpredict/MobilenetV1/MobilenetV1/Conv2d_10_depthwise/BatchNorm/FusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2;
9predict/MobilenetV1/MobilenetV1/Conv2d_10_depthwise/Relu6?
Ipredict/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/Conv2D/ReadVariableOpReadVariableOprtrain_mobilenetv1_mobilenetv1_conv2d_10_pointwise_kernel_regularizer_l2_regularizer_l2loss_readvariableop_resource*(
_output_shapes
:??*
dtype02K
Ipredict/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/Conv2D/ReadVariableOp?
:predict/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/Conv2DConv2DGpredict/MobilenetV1/MobilenetV1/Conv2d_10_depthwise/Relu6:activations:0Qpredict/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
2<
:predict/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/Conv2D?
Lpredict/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/BatchNorm/ReadVariableOpReadVariableOpStrain_mobilenetv1_mobilenetv1_conv2d_10_pointwise_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02N
Lpredict/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/BatchNorm/ReadVariableOp?
Npredict/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/BatchNorm/ReadVariableOp_1ReadVariableOpUtrain_mobilenetv1_mobilenetv1_conv2d_10_pointwise_batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype02P
Npredict/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/BatchNorm/ReadVariableOp_1?
]predict/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOpReadVariableOpdtrain_mobilenetv1_mobilenetv1_conv2d_10_pointwise_batchnorm_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02_
]predict/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp?
_predict/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpftrain_mobilenetv1_mobilenetv1_conv2d_10_pointwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02a
_predict/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1?
Npredict/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/BatchNorm/FusedBatchNormV3FusedBatchNormV3Cpredict/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/Conv2D:output:0Tpredict/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/BatchNorm/ReadVariableOp:value:0Vpredict/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/BatchNorm/ReadVariableOp_1:value:0epredict/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp:value:0gpredict/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( 2P
Npredict/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/BatchNorm/FusedBatchNormV3?
9predict/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/Relu6Relu6Rpredict/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/BatchNorm/FusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2;
9predict/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/Relu6?
Lpredict/MobilenetV1/MobilenetV1/Conv2d_11_depthwise/depthwise/ReadVariableOpReadVariableOpStrain_mobilenetv1_mobilenetv1_conv2d_11_depthwise_depthwise_readvariableop_resource*'
_output_shapes
:?*
dtype02N
Lpredict/MobilenetV1/MobilenetV1/Conv2d_11_depthwise/depthwise/ReadVariableOp?
=predict/MobilenetV1/MobilenetV1/Conv2d_11_depthwise/depthwiseDepthwiseConv2dNativeGpredict/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/Relu6:activations:0Tpredict/MobilenetV1/MobilenetV1/Conv2d_11_depthwise/depthwise/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
2?
=predict/MobilenetV1/MobilenetV1/Conv2d_11_depthwise/depthwise?
Lpredict/MobilenetV1/MobilenetV1/Conv2d_11_depthwise/BatchNorm/ReadVariableOpReadVariableOpStrain_mobilenetv1_mobilenetv1_conv2d_11_depthwise_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02N
Lpredict/MobilenetV1/MobilenetV1/Conv2d_11_depthwise/BatchNorm/ReadVariableOp?
Npredict/MobilenetV1/MobilenetV1/Conv2d_11_depthwise/BatchNorm/ReadVariableOp_1ReadVariableOpUtrain_mobilenetv1_mobilenetv1_conv2d_11_depthwise_batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype02P
Npredict/MobilenetV1/MobilenetV1/Conv2d_11_depthwise/BatchNorm/ReadVariableOp_1?
]predict/MobilenetV1/MobilenetV1/Conv2d_11_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOpReadVariableOpdtrain_mobilenetv1_mobilenetv1_conv2d_11_depthwise_batchnorm_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02_
]predict/MobilenetV1/MobilenetV1/Conv2d_11_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp?
_predict/MobilenetV1/MobilenetV1/Conv2d_11_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpftrain_mobilenetv1_mobilenetv1_conv2d_11_depthwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02a
_predict/MobilenetV1/MobilenetV1/Conv2d_11_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1?
Npredict/MobilenetV1/MobilenetV1/Conv2d_11_depthwise/BatchNorm/FusedBatchNormV3FusedBatchNormV3Fpredict/MobilenetV1/MobilenetV1/Conv2d_11_depthwise/depthwise:output:0Tpredict/MobilenetV1/MobilenetV1/Conv2d_11_depthwise/BatchNorm/ReadVariableOp:value:0Vpredict/MobilenetV1/MobilenetV1/Conv2d_11_depthwise/BatchNorm/ReadVariableOp_1:value:0epredict/MobilenetV1/MobilenetV1/Conv2d_11_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp:value:0gpredict/MobilenetV1/MobilenetV1/Conv2d_11_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( 2P
Npredict/MobilenetV1/MobilenetV1/Conv2d_11_depthwise/BatchNorm/FusedBatchNormV3?
9predict/MobilenetV1/MobilenetV1/Conv2d_11_depthwise/Relu6Relu6Rpredict/MobilenetV1/MobilenetV1/Conv2d_11_depthwise/BatchNorm/FusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2;
9predict/MobilenetV1/MobilenetV1/Conv2d_11_depthwise/Relu6?
Ipredict/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/Conv2D/ReadVariableOpReadVariableOprtrain_mobilenetv1_mobilenetv1_conv2d_11_pointwise_kernel_regularizer_l2_regularizer_l2loss_readvariableop_resource*(
_output_shapes
:??*
dtype02K
Ipredict/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/Conv2D/ReadVariableOp?
:predict/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/Conv2DConv2DGpredict/MobilenetV1/MobilenetV1/Conv2d_11_depthwise/Relu6:activations:0Qpredict/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
2<
:predict/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/Conv2D?
Lpredict/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/BatchNorm/ReadVariableOpReadVariableOpStrain_mobilenetv1_mobilenetv1_conv2d_11_pointwise_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02N
Lpredict/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/BatchNorm/ReadVariableOp?
Npredict/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/BatchNorm/ReadVariableOp_1ReadVariableOpUtrain_mobilenetv1_mobilenetv1_conv2d_11_pointwise_batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype02P
Npredict/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/BatchNorm/ReadVariableOp_1?
]predict/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOpReadVariableOpdtrain_mobilenetv1_mobilenetv1_conv2d_11_pointwise_batchnorm_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02_
]predict/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp?
_predict/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpftrain_mobilenetv1_mobilenetv1_conv2d_11_pointwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02a
_predict/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1?
Npredict/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/BatchNorm/FusedBatchNormV3FusedBatchNormV3Cpredict/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/Conv2D:output:0Tpredict/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/BatchNorm/ReadVariableOp:value:0Vpredict/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/BatchNorm/ReadVariableOp_1:value:0epredict/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp:value:0gpredict/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( 2P
Npredict/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/BatchNorm/FusedBatchNormV3?
9predict/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/Relu6Relu6Rpredict/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/BatchNorm/FusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2;
9predict/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/Relu6?
Lpredict/MobilenetV1/MobilenetV1/Conv2d_12_depthwise/depthwise/ReadVariableOpReadVariableOpStrain_mobilenetv1_mobilenetv1_conv2d_12_depthwise_depthwise_readvariableop_resource*'
_output_shapes
:?*
dtype02N
Lpredict/MobilenetV1/MobilenetV1/Conv2d_12_depthwise/depthwise/ReadVariableOp?
=predict/MobilenetV1/MobilenetV1/Conv2d_12_depthwise/depthwiseDepthwiseConv2dNativeGpredict/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/Relu6:activations:0Tpredict/MobilenetV1/MobilenetV1/Conv2d_12_depthwise/depthwise/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
2?
=predict/MobilenetV1/MobilenetV1/Conv2d_12_depthwise/depthwise?
Lpredict/MobilenetV1/MobilenetV1/Conv2d_12_depthwise/BatchNorm/ReadVariableOpReadVariableOpStrain_mobilenetv1_mobilenetv1_conv2d_12_depthwise_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02N
Lpredict/MobilenetV1/MobilenetV1/Conv2d_12_depthwise/BatchNorm/ReadVariableOp?
Npredict/MobilenetV1/MobilenetV1/Conv2d_12_depthwise/BatchNorm/ReadVariableOp_1ReadVariableOpUtrain_mobilenetv1_mobilenetv1_conv2d_12_depthwise_batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype02P
Npredict/MobilenetV1/MobilenetV1/Conv2d_12_depthwise/BatchNorm/ReadVariableOp_1?
]predict/MobilenetV1/MobilenetV1/Conv2d_12_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOpReadVariableOpdtrain_mobilenetv1_mobilenetv1_conv2d_12_depthwise_batchnorm_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02_
]predict/MobilenetV1/MobilenetV1/Conv2d_12_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp?
_predict/MobilenetV1/MobilenetV1/Conv2d_12_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpftrain_mobilenetv1_mobilenetv1_conv2d_12_depthwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02a
_predict/MobilenetV1/MobilenetV1/Conv2d_12_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1?
Npredict/MobilenetV1/MobilenetV1/Conv2d_12_depthwise/BatchNorm/FusedBatchNormV3FusedBatchNormV3Fpredict/MobilenetV1/MobilenetV1/Conv2d_12_depthwise/depthwise:output:0Tpredict/MobilenetV1/MobilenetV1/Conv2d_12_depthwise/BatchNorm/ReadVariableOp:value:0Vpredict/MobilenetV1/MobilenetV1/Conv2d_12_depthwise/BatchNorm/ReadVariableOp_1:value:0epredict/MobilenetV1/MobilenetV1/Conv2d_12_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp:value:0gpredict/MobilenetV1/MobilenetV1/Conv2d_12_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( 2P
Npredict/MobilenetV1/MobilenetV1/Conv2d_12_depthwise/BatchNorm/FusedBatchNormV3?
9predict/MobilenetV1/MobilenetV1/Conv2d_12_depthwise/Relu6Relu6Rpredict/MobilenetV1/MobilenetV1/Conv2d_12_depthwise/BatchNorm/FusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2;
9predict/MobilenetV1/MobilenetV1/Conv2d_12_depthwise/Relu6?
Ipredict/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/Conv2D/ReadVariableOpReadVariableOprtrain_mobilenetv1_mobilenetv1_conv2d_12_pointwise_kernel_regularizer_l2_regularizer_l2loss_readvariableop_resource*(
_output_shapes
:??*
dtype02K
Ipredict/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/Conv2D/ReadVariableOp?
:predict/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/Conv2DConv2DGpredict/MobilenetV1/MobilenetV1/Conv2d_12_depthwise/Relu6:activations:0Qpredict/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
2<
:predict/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/Conv2D?
Lpredict/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/BatchNorm/ReadVariableOpReadVariableOpStrain_mobilenetv1_mobilenetv1_conv2d_12_pointwise_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02N
Lpredict/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/BatchNorm/ReadVariableOp?
Npredict/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/BatchNorm/ReadVariableOp_1ReadVariableOpUtrain_mobilenetv1_mobilenetv1_conv2d_12_pointwise_batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype02P
Npredict/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/BatchNorm/ReadVariableOp_1?
]predict/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOpReadVariableOpdtrain_mobilenetv1_mobilenetv1_conv2d_12_pointwise_batchnorm_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02_
]predict/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp?
_predict/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpftrain_mobilenetv1_mobilenetv1_conv2d_12_pointwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02a
_predict/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1?
Npredict/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/BatchNorm/FusedBatchNormV3FusedBatchNormV3Cpredict/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/Conv2D:output:0Tpredict/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/BatchNorm/ReadVariableOp:value:0Vpredict/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/BatchNorm/ReadVariableOp_1:value:0epredict/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp:value:0gpredict/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( 2P
Npredict/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/BatchNorm/FusedBatchNormV3?
9predict/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/Relu6Relu6Rpredict/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/BatchNorm/FusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2;
9predict/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/Relu6?
Lpredict/MobilenetV1/MobilenetV1/Conv2d_13_depthwise/depthwise/ReadVariableOpReadVariableOpStrain_mobilenetv1_mobilenetv1_conv2d_13_depthwise_depthwise_readvariableop_resource*'
_output_shapes
:?*
dtype02N
Lpredict/MobilenetV1/MobilenetV1/Conv2d_13_depthwise/depthwise/ReadVariableOp?
=predict/MobilenetV1/MobilenetV1/Conv2d_13_depthwise/depthwiseDepthwiseConv2dNativeGpredict/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/Relu6:activations:0Tpredict/MobilenetV1/MobilenetV1/Conv2d_13_depthwise/depthwise/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
2?
=predict/MobilenetV1/MobilenetV1/Conv2d_13_depthwise/depthwise?
Lpredict/MobilenetV1/MobilenetV1/Conv2d_13_depthwise/BatchNorm/ReadVariableOpReadVariableOpStrain_mobilenetv1_mobilenetv1_conv2d_13_depthwise_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02N
Lpredict/MobilenetV1/MobilenetV1/Conv2d_13_depthwise/BatchNorm/ReadVariableOp?
Npredict/MobilenetV1/MobilenetV1/Conv2d_13_depthwise/BatchNorm/ReadVariableOp_1ReadVariableOpUtrain_mobilenetv1_mobilenetv1_conv2d_13_depthwise_batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype02P
Npredict/MobilenetV1/MobilenetV1/Conv2d_13_depthwise/BatchNorm/ReadVariableOp_1?
]predict/MobilenetV1/MobilenetV1/Conv2d_13_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOpReadVariableOpdtrain_mobilenetv1_mobilenetv1_conv2d_13_depthwise_batchnorm_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02_
]predict/MobilenetV1/MobilenetV1/Conv2d_13_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp?
_predict/MobilenetV1/MobilenetV1/Conv2d_13_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpftrain_mobilenetv1_mobilenetv1_conv2d_13_depthwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02a
_predict/MobilenetV1/MobilenetV1/Conv2d_13_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1?
Npredict/MobilenetV1/MobilenetV1/Conv2d_13_depthwise/BatchNorm/FusedBatchNormV3FusedBatchNormV3Fpredict/MobilenetV1/MobilenetV1/Conv2d_13_depthwise/depthwise:output:0Tpredict/MobilenetV1/MobilenetV1/Conv2d_13_depthwise/BatchNorm/ReadVariableOp:value:0Vpredict/MobilenetV1/MobilenetV1/Conv2d_13_depthwise/BatchNorm/ReadVariableOp_1:value:0epredict/MobilenetV1/MobilenetV1/Conv2d_13_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp:value:0gpredict/MobilenetV1/MobilenetV1/Conv2d_13_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( 2P
Npredict/MobilenetV1/MobilenetV1/Conv2d_13_depthwise/BatchNorm/FusedBatchNormV3?
9predict/MobilenetV1/MobilenetV1/Conv2d_13_depthwise/Relu6Relu6Rpredict/MobilenetV1/MobilenetV1/Conv2d_13_depthwise/BatchNorm/FusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2;
9predict/MobilenetV1/MobilenetV1/Conv2d_13_depthwise/Relu6?
Ipredict/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/Conv2D/ReadVariableOpReadVariableOprtrain_mobilenetv1_mobilenetv1_conv2d_13_pointwise_kernel_regularizer_l2_regularizer_l2loss_readvariableop_resource*(
_output_shapes
:??*
dtype02K
Ipredict/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/Conv2D/ReadVariableOp?
:predict/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/Conv2DConv2DGpredict/MobilenetV1/MobilenetV1/Conv2d_13_depthwise/Relu6:activations:0Qpredict/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
2<
:predict/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/Conv2D?
Lpredict/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/BatchNorm/ReadVariableOpReadVariableOpStrain_mobilenetv1_mobilenetv1_conv2d_13_pointwise_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02N
Lpredict/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/BatchNorm/ReadVariableOp?
Npredict/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/BatchNorm/ReadVariableOp_1ReadVariableOpUtrain_mobilenetv1_mobilenetv1_conv2d_13_pointwise_batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype02P
Npredict/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/BatchNorm/ReadVariableOp_1?
]predict/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOpReadVariableOpdtrain_mobilenetv1_mobilenetv1_conv2d_13_pointwise_batchnorm_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02_
]predict/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp?
_predict/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpftrain_mobilenetv1_mobilenetv1_conv2d_13_pointwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02a
_predict/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1?
Npredict/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/BatchNorm/FusedBatchNormV3FusedBatchNormV3Cpredict/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/Conv2D:output:0Tpredict/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/BatchNorm/ReadVariableOp:value:0Vpredict/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/BatchNorm/ReadVariableOp_1:value:0epredict/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp:value:0gpredict/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( 2P
Npredict/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/BatchNorm/FusedBatchNormV3?
9predict/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/Relu6Relu6Rpredict/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/BatchNorm/FusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2;
9predict/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/Relu6?
8predict/MobilenetV1/Logits/global_pool/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2:
8predict/MobilenetV1/Logits/global_pool/reduction_indices?
&predict/MobilenetV1/Logits/global_poolMeanGpredict/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/Relu6:activations:0Apredict/MobilenetV1/Logits/global_pool/reduction_indices:output:0*
T0*0
_output_shapes
:??????????*
	keep_dims(2(
&predict/MobilenetV1/Logits/global_pool?
.predict/MobilenetV1/Logits/Dropout_1b/IdentityIdentity/predict/MobilenetV1/Logits/global_pool:output:0*
T0*0
_output_shapes
:??????????20
.predict/MobilenetV1/Logits/Dropout_1b/Identity?
>predict/MobilenetV1/Logits/Conv2d_1c_1x1/Conv2D/ReadVariableOpReadVariableOpgtrain_mobilenetv1_logits_conv2d_1c_1x1_kernel_regularizer_l2_regularizer_l2loss_readvariableop_resource*(
_output_shapes
:??*
dtype02@
>predict/MobilenetV1/Logits/Conv2d_1c_1x1/Conv2D/ReadVariableOp?
/predict/MobilenetV1/Logits/Conv2d_1c_1x1/Conv2DConv2D7predict/MobilenetV1/Logits/Dropout_1b/Identity:output:0Fpredict/MobilenetV1/Logits/Conv2d_1c_1x1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
21
/predict/MobilenetV1/Logits/Conv2d_1c_1x1/Conv2D?
?predict/MobilenetV1/Logits/Conv2d_1c_1x1/BiasAdd/ReadVariableOpReadVariableOpFtrain_mobilenetv1_logits_conv2d_1c_1x1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02A
?predict/MobilenetV1/Logits/Conv2d_1c_1x1/BiasAdd/ReadVariableOp?
0predict/MobilenetV1/Logits/Conv2d_1c_1x1/BiasAddBiasAdd8predict/MobilenetV1/Logits/Conv2d_1c_1x1/Conv2D:output:0Gpredict/MobilenetV1/Logits/Conv2d_1c_1x1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????22
0predict/MobilenetV1/Logits/Conv2d_1c_1x1/BiasAdd?
)predict/MobilenetV1/Logits/SpatialSqueezeSqueeze9predict/MobilenetV1/Logits/Conv2d_1c_1x1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????*
squeeze_dims
2+
)predict/MobilenetV1/Logits/SpatialSqueeze?
-predict/MobilenetV1/Predictions/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"?????  2/
-predict/MobilenetV1/Predictions/Reshape/shape?
'predict/MobilenetV1/Predictions/ReshapeReshape2predict/MobilenetV1/Logits/SpatialSqueeze:output:06predict/MobilenetV1/Predictions/Reshape/shape:output:0*
T0*(
_output_shapes
:??????????2)
'predict/MobilenetV1/Predictions/Reshape?
'predict/MobilenetV1/Predictions/SoftmaxSoftmax0predict/MobilenetV1/Predictions/Reshape:output:0*
T0*(
_output_shapes
:??????????2)
'predict/MobilenetV1/Predictions/Softmax?
%predict/MobilenetV1/Predictions/ShapeShape2predict/MobilenetV1/Logits/SpatialSqueeze:output:0*
T0*
_output_shapes
:2'
%predict/MobilenetV1/Predictions/Shape?
)predict/MobilenetV1/Predictions/Reshape_1Reshape1predict/MobilenetV1/Predictions/Softmax:softmax:0.predict/MobilenetV1/Predictions/Shape:output:0*
T0*(
_output_shapes
:??????????2+
)predict/MobilenetV1/Predictions/Reshape_1"Y
&predict_mobilenetv1_logits_global_pool/predict/MobilenetV1/Logits/global_pool:output:0"_
)predict_mobilenetv1_logits_spatialsqueeze2predict/MobilenetV1/Logits/SpatialSqueeze:output:0"a
+predict_mobilenetv1_logits_spatialsqueeze_02predict/MobilenetV1/Logits/SpatialSqueeze:output:0"n
.predict_mobilenetv1_mobilenetv1_conv2d_0_relu6<predict/MobilenetV1/MobilenetV1/Conv2d_0/Relu6:activations:0"?
9predict_mobilenetv1_mobilenetv1_conv2d_10_depthwise_relu6Gpredict/MobilenetV1/MobilenetV1/Conv2d_10_depthwise/Relu6:activations:0"?
9predict_mobilenetv1_mobilenetv1_conv2d_10_pointwise_relu6Gpredict/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/Relu6:activations:0"?
9predict_mobilenetv1_mobilenetv1_conv2d_11_depthwise_relu6Gpredict/MobilenetV1/MobilenetV1/Conv2d_11_depthwise/Relu6:activations:0"?
9predict_mobilenetv1_mobilenetv1_conv2d_11_pointwise_relu6Gpredict/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/Relu6:activations:0"?
9predict_mobilenetv1_mobilenetv1_conv2d_12_depthwise_relu6Gpredict/MobilenetV1/MobilenetV1/Conv2d_12_depthwise/Relu6:activations:0"?
9predict_mobilenetv1_mobilenetv1_conv2d_12_pointwise_relu6Gpredict/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/Relu6:activations:0"?
9predict_mobilenetv1_mobilenetv1_conv2d_13_depthwise_relu6Gpredict/MobilenetV1/MobilenetV1/Conv2d_13_depthwise/Relu6:activations:0"?
9predict_mobilenetv1_mobilenetv1_conv2d_13_pointwise_relu6Gpredict/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/Relu6:activations:0"?
8predict_mobilenetv1_mobilenetv1_conv2d_1_depthwise_relu6Fpredict/MobilenetV1/MobilenetV1/Conv2d_1_depthwise/Relu6:activations:0"?
8predict_mobilenetv1_mobilenetv1_conv2d_1_pointwise_relu6Fpredict/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/Relu6:activations:0"?
8predict_mobilenetv1_mobilenetv1_conv2d_2_depthwise_relu6Fpredict/MobilenetV1/MobilenetV1/Conv2d_2_depthwise/Relu6:activations:0"?
8predict_mobilenetv1_mobilenetv1_conv2d_2_pointwise_relu6Fpredict/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/Relu6:activations:0"?
8predict_mobilenetv1_mobilenetv1_conv2d_3_depthwise_relu6Fpredict/MobilenetV1/MobilenetV1/Conv2d_3_depthwise/Relu6:activations:0"?
8predict_mobilenetv1_mobilenetv1_conv2d_3_pointwise_relu6Fpredict/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/Relu6:activations:0"?
8predict_mobilenetv1_mobilenetv1_conv2d_4_depthwise_relu6Fpredict/MobilenetV1/MobilenetV1/Conv2d_4_depthwise/Relu6:activations:0"?
8predict_mobilenetv1_mobilenetv1_conv2d_4_pointwise_relu6Fpredict/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/Relu6:activations:0"?
8predict_mobilenetv1_mobilenetv1_conv2d_5_depthwise_relu6Fpredict/MobilenetV1/MobilenetV1/Conv2d_5_depthwise/Relu6:activations:0"?
8predict_mobilenetv1_mobilenetv1_conv2d_5_pointwise_relu6Fpredict/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/Relu6:activations:0"?
8predict_mobilenetv1_mobilenetv1_conv2d_6_depthwise_relu6Fpredict/MobilenetV1/MobilenetV1/Conv2d_6_depthwise/Relu6:activations:0"?
8predict_mobilenetv1_mobilenetv1_conv2d_6_pointwise_relu6Fpredict/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/Relu6:activations:0"?
8predict_mobilenetv1_mobilenetv1_conv2d_7_depthwise_relu6Fpredict/MobilenetV1/MobilenetV1/Conv2d_7_depthwise/Relu6:activations:0"?
8predict_mobilenetv1_mobilenetv1_conv2d_7_pointwise_relu6Fpredict/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/Relu6:activations:0"?
8predict_mobilenetv1_mobilenetv1_conv2d_8_depthwise_relu6Fpredict/MobilenetV1/MobilenetV1/Conv2d_8_depthwise/Relu6:activations:0"?
8predict_mobilenetv1_mobilenetv1_conv2d_8_pointwise_relu6Fpredict/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/Relu6:activations:0"?
8predict_mobilenetv1_mobilenetv1_conv2d_9_depthwise_relu6Fpredict/MobilenetV1/MobilenetV1/Conv2d_9_depthwise/Relu6:activations:0"?
8predict_mobilenetv1_mobilenetv1_conv2d_9_pointwise_relu6Fpredict/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/Relu6:activations:0"_
)predict_mobilenetv1_predictions_reshape_12predict/MobilenetV1/Predictions/Reshape_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:+???????????????????????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : :G C
A
_output_shapes/
-:+???????????????????????????:

_output_shapes
: 
??
??
__inference_pruned_2884
placeholder
placeholder_2?
gtrain_mobilenetv1_mobilenetv1_conv2d_0_kernel_regularizer_l2_regularizer_l2loss_readvariableop_resource:V
Htrain_mobilenetv1_mobilenetv1_conv2d_0_batchnorm_readvariableop_resource:X
Jtrain_mobilenetv1_mobilenetv1_conv2d_0_batchnorm_readvariableop_1_resource:g
Ytrain_mobilenetv1_mobilenetv1_conv2d_0_batchnorm_fusedbatchnormv3_readvariableop_resource:i
[train_mobilenetv1_mobilenetv1_conv2d_0_batchnorm_fusedbatchnormv3_readvariableop_1_resource:l
Rtrain_mobilenetv1_mobilenetv1_conv2d_1_depthwise_depthwise_readvariableop_resource:`
Rtrain_mobilenetv1_mobilenetv1_conv2d_1_depthwise_batchnorm_readvariableop_resource:b
Ttrain_mobilenetv1_mobilenetv1_conv2d_1_depthwise_batchnorm_readvariableop_1_resource:q
ctrain_mobilenetv1_mobilenetv1_conv2d_1_depthwise_batchnorm_fusedbatchnormv3_readvariableop_resource:s
etrain_mobilenetv1_mobilenetv1_conv2d_1_depthwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource:?
qtrain_mobilenetv1_mobilenetv1_conv2d_1_pointwise_kernel_regularizer_l2_regularizer_l2loss_readvariableop_resource:0`
Rtrain_mobilenetv1_mobilenetv1_conv2d_1_pointwise_batchnorm_readvariableop_resource:0b
Ttrain_mobilenetv1_mobilenetv1_conv2d_1_pointwise_batchnorm_readvariableop_1_resource:0q
ctrain_mobilenetv1_mobilenetv1_conv2d_1_pointwise_batchnorm_fusedbatchnormv3_readvariableop_resource:0s
etrain_mobilenetv1_mobilenetv1_conv2d_1_pointwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource:0l
Rtrain_mobilenetv1_mobilenetv1_conv2d_2_depthwise_depthwise_readvariableop_resource:0`
Rtrain_mobilenetv1_mobilenetv1_conv2d_2_depthwise_batchnorm_readvariableop_resource:0b
Ttrain_mobilenetv1_mobilenetv1_conv2d_2_depthwise_batchnorm_readvariableop_1_resource:0q
ctrain_mobilenetv1_mobilenetv1_conv2d_2_depthwise_batchnorm_fusedbatchnormv3_readvariableop_resource:0s
etrain_mobilenetv1_mobilenetv1_conv2d_2_depthwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource:0?
qtrain_mobilenetv1_mobilenetv1_conv2d_2_pointwise_kernel_regularizer_l2_regularizer_l2loss_readvariableop_resource:0``
Rtrain_mobilenetv1_mobilenetv1_conv2d_2_pointwise_batchnorm_readvariableop_resource:`b
Ttrain_mobilenetv1_mobilenetv1_conv2d_2_pointwise_batchnorm_readvariableop_1_resource:`q
ctrain_mobilenetv1_mobilenetv1_conv2d_2_pointwise_batchnorm_fusedbatchnormv3_readvariableop_resource:`s
etrain_mobilenetv1_mobilenetv1_conv2d_2_pointwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource:`l
Rtrain_mobilenetv1_mobilenetv1_conv2d_3_depthwise_depthwise_readvariableop_resource:``
Rtrain_mobilenetv1_mobilenetv1_conv2d_3_depthwise_batchnorm_readvariableop_resource:`b
Ttrain_mobilenetv1_mobilenetv1_conv2d_3_depthwise_batchnorm_readvariableop_1_resource:`q
ctrain_mobilenetv1_mobilenetv1_conv2d_3_depthwise_batchnorm_fusedbatchnormv3_readvariableop_resource:`s
etrain_mobilenetv1_mobilenetv1_conv2d_3_depthwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource:`?
qtrain_mobilenetv1_mobilenetv1_conv2d_3_pointwise_kernel_regularizer_l2_regularizer_l2loss_readvariableop_resource:```
Rtrain_mobilenetv1_mobilenetv1_conv2d_3_pointwise_batchnorm_readvariableop_resource:`b
Ttrain_mobilenetv1_mobilenetv1_conv2d_3_pointwise_batchnorm_readvariableop_1_resource:`q
ctrain_mobilenetv1_mobilenetv1_conv2d_3_pointwise_batchnorm_fusedbatchnormv3_readvariableop_resource:`s
etrain_mobilenetv1_mobilenetv1_conv2d_3_pointwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource:`l
Rtrain_mobilenetv1_mobilenetv1_conv2d_4_depthwise_depthwise_readvariableop_resource:``
Rtrain_mobilenetv1_mobilenetv1_conv2d_4_depthwise_batchnorm_readvariableop_resource:`b
Ttrain_mobilenetv1_mobilenetv1_conv2d_4_depthwise_batchnorm_readvariableop_1_resource:`q
ctrain_mobilenetv1_mobilenetv1_conv2d_4_depthwise_batchnorm_fusedbatchnormv3_readvariableop_resource:`s
etrain_mobilenetv1_mobilenetv1_conv2d_4_depthwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource:`?
qtrain_mobilenetv1_mobilenetv1_conv2d_4_pointwise_kernel_regularizer_l2_regularizer_l2loss_readvariableop_resource:`?a
Rtrain_mobilenetv1_mobilenetv1_conv2d_4_pointwise_batchnorm_readvariableop_resource:	?c
Ttrain_mobilenetv1_mobilenetv1_conv2d_4_pointwise_batchnorm_readvariableop_1_resource:	?r
ctrain_mobilenetv1_mobilenetv1_conv2d_4_pointwise_batchnorm_fusedbatchnormv3_readvariableop_resource:	?t
etrain_mobilenetv1_mobilenetv1_conv2d_4_pointwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource:	?m
Rtrain_mobilenetv1_mobilenetv1_conv2d_5_depthwise_depthwise_readvariableop_resource:?a
Rtrain_mobilenetv1_mobilenetv1_conv2d_5_depthwise_batchnorm_readvariableop_resource:	?c
Ttrain_mobilenetv1_mobilenetv1_conv2d_5_depthwise_batchnorm_readvariableop_1_resource:	?r
ctrain_mobilenetv1_mobilenetv1_conv2d_5_depthwise_batchnorm_fusedbatchnormv3_readvariableop_resource:	?t
etrain_mobilenetv1_mobilenetv1_conv2d_5_depthwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource:	??
qtrain_mobilenetv1_mobilenetv1_conv2d_5_pointwise_kernel_regularizer_l2_regularizer_l2loss_readvariableop_resource:??a
Rtrain_mobilenetv1_mobilenetv1_conv2d_5_pointwise_batchnorm_readvariableop_resource:	?c
Ttrain_mobilenetv1_mobilenetv1_conv2d_5_pointwise_batchnorm_readvariableop_1_resource:	?r
ctrain_mobilenetv1_mobilenetv1_conv2d_5_pointwise_batchnorm_fusedbatchnormv3_readvariableop_resource:	?t
etrain_mobilenetv1_mobilenetv1_conv2d_5_pointwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource:	?m
Rtrain_mobilenetv1_mobilenetv1_conv2d_6_depthwise_depthwise_readvariableop_resource:?a
Rtrain_mobilenetv1_mobilenetv1_conv2d_6_depthwise_batchnorm_readvariableop_resource:	?c
Ttrain_mobilenetv1_mobilenetv1_conv2d_6_depthwise_batchnorm_readvariableop_1_resource:	?r
ctrain_mobilenetv1_mobilenetv1_conv2d_6_depthwise_batchnorm_fusedbatchnormv3_readvariableop_resource:	?t
etrain_mobilenetv1_mobilenetv1_conv2d_6_depthwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource:	??
qtrain_mobilenetv1_mobilenetv1_conv2d_6_pointwise_kernel_regularizer_l2_regularizer_l2loss_readvariableop_resource:??a
Rtrain_mobilenetv1_mobilenetv1_conv2d_6_pointwise_batchnorm_readvariableop_resource:	?c
Ttrain_mobilenetv1_mobilenetv1_conv2d_6_pointwise_batchnorm_readvariableop_1_resource:	?r
ctrain_mobilenetv1_mobilenetv1_conv2d_6_pointwise_batchnorm_fusedbatchnormv3_readvariableop_resource:	?t
etrain_mobilenetv1_mobilenetv1_conv2d_6_pointwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource:	?m
Rtrain_mobilenetv1_mobilenetv1_conv2d_7_depthwise_depthwise_readvariableop_resource:?a
Rtrain_mobilenetv1_mobilenetv1_conv2d_7_depthwise_batchnorm_readvariableop_resource:	?c
Ttrain_mobilenetv1_mobilenetv1_conv2d_7_depthwise_batchnorm_readvariableop_1_resource:	?r
ctrain_mobilenetv1_mobilenetv1_conv2d_7_depthwise_batchnorm_fusedbatchnormv3_readvariableop_resource:	?t
etrain_mobilenetv1_mobilenetv1_conv2d_7_depthwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource:	??
qtrain_mobilenetv1_mobilenetv1_conv2d_7_pointwise_kernel_regularizer_l2_regularizer_l2loss_readvariableop_resource:??a
Rtrain_mobilenetv1_mobilenetv1_conv2d_7_pointwise_batchnorm_readvariableop_resource:	?c
Ttrain_mobilenetv1_mobilenetv1_conv2d_7_pointwise_batchnorm_readvariableop_1_resource:	?r
ctrain_mobilenetv1_mobilenetv1_conv2d_7_pointwise_batchnorm_fusedbatchnormv3_readvariableop_resource:	?t
etrain_mobilenetv1_mobilenetv1_conv2d_7_pointwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource:	?m
Rtrain_mobilenetv1_mobilenetv1_conv2d_8_depthwise_depthwise_readvariableop_resource:?a
Rtrain_mobilenetv1_mobilenetv1_conv2d_8_depthwise_batchnorm_readvariableop_resource:	?c
Ttrain_mobilenetv1_mobilenetv1_conv2d_8_depthwise_batchnorm_readvariableop_1_resource:	?r
ctrain_mobilenetv1_mobilenetv1_conv2d_8_depthwise_batchnorm_fusedbatchnormv3_readvariableop_resource:	?t
etrain_mobilenetv1_mobilenetv1_conv2d_8_depthwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource:	??
qtrain_mobilenetv1_mobilenetv1_conv2d_8_pointwise_kernel_regularizer_l2_regularizer_l2loss_readvariableop_resource:??a
Rtrain_mobilenetv1_mobilenetv1_conv2d_8_pointwise_batchnorm_readvariableop_resource:	?c
Ttrain_mobilenetv1_mobilenetv1_conv2d_8_pointwise_batchnorm_readvariableop_1_resource:	?r
ctrain_mobilenetv1_mobilenetv1_conv2d_8_pointwise_batchnorm_fusedbatchnormv3_readvariableop_resource:	?t
etrain_mobilenetv1_mobilenetv1_conv2d_8_pointwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource:	?m
Rtrain_mobilenetv1_mobilenetv1_conv2d_9_depthwise_depthwise_readvariableop_resource:?a
Rtrain_mobilenetv1_mobilenetv1_conv2d_9_depthwise_batchnorm_readvariableop_resource:	?c
Ttrain_mobilenetv1_mobilenetv1_conv2d_9_depthwise_batchnorm_readvariableop_1_resource:	?r
ctrain_mobilenetv1_mobilenetv1_conv2d_9_depthwise_batchnorm_fusedbatchnormv3_readvariableop_resource:	?t
etrain_mobilenetv1_mobilenetv1_conv2d_9_depthwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource:	??
qtrain_mobilenetv1_mobilenetv1_conv2d_9_pointwise_kernel_regularizer_l2_regularizer_l2loss_readvariableop_resource:??a
Rtrain_mobilenetv1_mobilenetv1_conv2d_9_pointwise_batchnorm_readvariableop_resource:	?c
Ttrain_mobilenetv1_mobilenetv1_conv2d_9_pointwise_batchnorm_readvariableop_1_resource:	?r
ctrain_mobilenetv1_mobilenetv1_conv2d_9_pointwise_batchnorm_fusedbatchnormv3_readvariableop_resource:	?t
etrain_mobilenetv1_mobilenetv1_conv2d_9_pointwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource:	?n
Strain_mobilenetv1_mobilenetv1_conv2d_10_depthwise_depthwise_readvariableop_resource:?b
Strain_mobilenetv1_mobilenetv1_conv2d_10_depthwise_batchnorm_readvariableop_resource:	?d
Utrain_mobilenetv1_mobilenetv1_conv2d_10_depthwise_batchnorm_readvariableop_1_resource:	?s
dtrain_mobilenetv1_mobilenetv1_conv2d_10_depthwise_batchnorm_fusedbatchnormv3_readvariableop_resource:	?u
ftrain_mobilenetv1_mobilenetv1_conv2d_10_depthwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource:	??
rtrain_mobilenetv1_mobilenetv1_conv2d_10_pointwise_kernel_regularizer_l2_regularizer_l2loss_readvariableop_resource:??b
Strain_mobilenetv1_mobilenetv1_conv2d_10_pointwise_batchnorm_readvariableop_resource:	?d
Utrain_mobilenetv1_mobilenetv1_conv2d_10_pointwise_batchnorm_readvariableop_1_resource:	?s
dtrain_mobilenetv1_mobilenetv1_conv2d_10_pointwise_batchnorm_fusedbatchnormv3_readvariableop_resource:	?u
ftrain_mobilenetv1_mobilenetv1_conv2d_10_pointwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource:	?n
Strain_mobilenetv1_mobilenetv1_conv2d_11_depthwise_depthwise_readvariableop_resource:?b
Strain_mobilenetv1_mobilenetv1_conv2d_11_depthwise_batchnorm_readvariableop_resource:	?d
Utrain_mobilenetv1_mobilenetv1_conv2d_11_depthwise_batchnorm_readvariableop_1_resource:	?s
dtrain_mobilenetv1_mobilenetv1_conv2d_11_depthwise_batchnorm_fusedbatchnormv3_readvariableop_resource:	?u
ftrain_mobilenetv1_mobilenetv1_conv2d_11_depthwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource:	??
rtrain_mobilenetv1_mobilenetv1_conv2d_11_pointwise_kernel_regularizer_l2_regularizer_l2loss_readvariableop_resource:??b
Strain_mobilenetv1_mobilenetv1_conv2d_11_pointwise_batchnorm_readvariableop_resource:	?d
Utrain_mobilenetv1_mobilenetv1_conv2d_11_pointwise_batchnorm_readvariableop_1_resource:	?s
dtrain_mobilenetv1_mobilenetv1_conv2d_11_pointwise_batchnorm_fusedbatchnormv3_readvariableop_resource:	?u
ftrain_mobilenetv1_mobilenetv1_conv2d_11_pointwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource:	?n
Strain_mobilenetv1_mobilenetv1_conv2d_12_depthwise_depthwise_readvariableop_resource:?b
Strain_mobilenetv1_mobilenetv1_conv2d_12_depthwise_batchnorm_readvariableop_resource:	?d
Utrain_mobilenetv1_mobilenetv1_conv2d_12_depthwise_batchnorm_readvariableop_1_resource:	?s
dtrain_mobilenetv1_mobilenetv1_conv2d_12_depthwise_batchnorm_fusedbatchnormv3_readvariableop_resource:	?u
ftrain_mobilenetv1_mobilenetv1_conv2d_12_depthwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource:	??
rtrain_mobilenetv1_mobilenetv1_conv2d_12_pointwise_kernel_regularizer_l2_regularizer_l2loss_readvariableop_resource:??b
Strain_mobilenetv1_mobilenetv1_conv2d_12_pointwise_batchnorm_readvariableop_resource:	?d
Utrain_mobilenetv1_mobilenetv1_conv2d_12_pointwise_batchnorm_readvariableop_1_resource:	?s
dtrain_mobilenetv1_mobilenetv1_conv2d_12_pointwise_batchnorm_fusedbatchnormv3_readvariableop_resource:	?u
ftrain_mobilenetv1_mobilenetv1_conv2d_12_pointwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource:	?n
Strain_mobilenetv1_mobilenetv1_conv2d_13_depthwise_depthwise_readvariableop_resource:?b
Strain_mobilenetv1_mobilenetv1_conv2d_13_depthwise_batchnorm_readvariableop_resource:	?d
Utrain_mobilenetv1_mobilenetv1_conv2d_13_depthwise_batchnorm_readvariableop_1_resource:	?s
dtrain_mobilenetv1_mobilenetv1_conv2d_13_depthwise_batchnorm_fusedbatchnormv3_readvariableop_resource:	?u
ftrain_mobilenetv1_mobilenetv1_conv2d_13_depthwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource:	??
rtrain_mobilenetv1_mobilenetv1_conv2d_13_pointwise_kernel_regularizer_l2_regularizer_l2loss_readvariableop_resource:??b
Strain_mobilenetv1_mobilenetv1_conv2d_13_pointwise_batchnorm_readvariableop_resource:	?d
Utrain_mobilenetv1_mobilenetv1_conv2d_13_pointwise_batchnorm_readvariableop_1_resource:	?s
dtrain_mobilenetv1_mobilenetv1_conv2d_13_pointwise_batchnorm_fusedbatchnormv3_readvariableop_resource:	?u
ftrain_mobilenetv1_mobilenetv1_conv2d_13_pointwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource:	??
gtrain_mobilenetv1_logits_conv2d_1c_1x1_kernel_regularizer_l2_regularizer_l2loss_readvariableop_resource:??U
Ftrain_mobilenetv1_logits_conv2d_1c_1x1_biasadd_readvariableop_resource:	?+
'train_mobilenetv1_logits_spatialsqueeze0
,train_mobilenetv1_mobilenetv1_conv2d_0_relu6;
7train_mobilenetv1_mobilenetv1_conv2d_10_depthwise_relu6;
7train_mobilenetv1_mobilenetv1_conv2d_10_pointwise_relu6;
7train_mobilenetv1_mobilenetv1_conv2d_11_depthwise_relu6;
7train_mobilenetv1_mobilenetv1_conv2d_11_pointwise_relu6;
7train_mobilenetv1_mobilenetv1_conv2d_12_depthwise_relu6;
7train_mobilenetv1_mobilenetv1_conv2d_12_pointwise_relu6;
7train_mobilenetv1_mobilenetv1_conv2d_13_depthwise_relu6;
7train_mobilenetv1_mobilenetv1_conv2d_13_pointwise_relu6:
6train_mobilenetv1_mobilenetv1_conv2d_1_depthwise_relu6:
6train_mobilenetv1_mobilenetv1_conv2d_1_pointwise_relu6:
6train_mobilenetv1_mobilenetv1_conv2d_2_depthwise_relu6:
6train_mobilenetv1_mobilenetv1_conv2d_2_pointwise_relu6:
6train_mobilenetv1_mobilenetv1_conv2d_3_depthwise_relu6:
6train_mobilenetv1_mobilenetv1_conv2d_3_pointwise_relu6:
6train_mobilenetv1_mobilenetv1_conv2d_4_depthwise_relu6:
6train_mobilenetv1_mobilenetv1_conv2d_4_pointwise_relu6:
6train_mobilenetv1_mobilenetv1_conv2d_5_depthwise_relu6:
6train_mobilenetv1_mobilenetv1_conv2d_5_pointwise_relu6:
6train_mobilenetv1_mobilenetv1_conv2d_6_depthwise_relu6:
6train_mobilenetv1_mobilenetv1_conv2d_6_pointwise_relu6:
6train_mobilenetv1_mobilenetv1_conv2d_7_depthwise_relu6:
6train_mobilenetv1_mobilenetv1_conv2d_7_pointwise_relu6:
6train_mobilenetv1_mobilenetv1_conv2d_8_depthwise_relu6:
6train_mobilenetv1_mobilenetv1_conv2d_8_pointwise_relu6:
6train_mobilenetv1_mobilenetv1_conv2d_9_depthwise_relu6:
6train_mobilenetv1_mobilenetv1_conv2d_9_pointwise_relu6-
)train_mobilenetv1_logits_spatialsqueeze_0+
'train_mobilenetv1_predictions_reshape_1(
$train_mobilenetv1_logits_global_pool??@train/MobilenetV1/MobilenetV1/Conv2d_0/BatchNorm/AssignMovingAvg?Btrain/MobilenetV1/MobilenetV1/Conv2d_0/BatchNorm/AssignMovingAvg_1?Ktrain/MobilenetV1/MobilenetV1/Conv2d_10_depthwise/BatchNorm/AssignMovingAvg?Mtrain/MobilenetV1/MobilenetV1/Conv2d_10_depthwise/BatchNorm/AssignMovingAvg_1?Ktrain/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/BatchNorm/AssignMovingAvg?Mtrain/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/BatchNorm/AssignMovingAvg_1?Ktrain/MobilenetV1/MobilenetV1/Conv2d_11_depthwise/BatchNorm/AssignMovingAvg?Mtrain/MobilenetV1/MobilenetV1/Conv2d_11_depthwise/BatchNorm/AssignMovingAvg_1?Ktrain/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/BatchNorm/AssignMovingAvg?Mtrain/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/BatchNorm/AssignMovingAvg_1?Ktrain/MobilenetV1/MobilenetV1/Conv2d_12_depthwise/BatchNorm/AssignMovingAvg?Mtrain/MobilenetV1/MobilenetV1/Conv2d_12_depthwise/BatchNorm/AssignMovingAvg_1?Ktrain/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/BatchNorm/AssignMovingAvg?Mtrain/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/BatchNorm/AssignMovingAvg_1?Ktrain/MobilenetV1/MobilenetV1/Conv2d_13_depthwise/BatchNorm/AssignMovingAvg?Mtrain/MobilenetV1/MobilenetV1/Conv2d_13_depthwise/BatchNorm/AssignMovingAvg_1?Ktrain/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/BatchNorm/AssignMovingAvg?Mtrain/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/BatchNorm/AssignMovingAvg_1?Jtrain/MobilenetV1/MobilenetV1/Conv2d_1_depthwise/BatchNorm/AssignMovingAvg?Ltrain/MobilenetV1/MobilenetV1/Conv2d_1_depthwise/BatchNorm/AssignMovingAvg_1?Jtrain/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/BatchNorm/AssignMovingAvg?Ltrain/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/BatchNorm/AssignMovingAvg_1?Jtrain/MobilenetV1/MobilenetV1/Conv2d_2_depthwise/BatchNorm/AssignMovingAvg?Ltrain/MobilenetV1/MobilenetV1/Conv2d_2_depthwise/BatchNorm/AssignMovingAvg_1?Jtrain/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/BatchNorm/AssignMovingAvg?Ltrain/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/BatchNorm/AssignMovingAvg_1?Jtrain/MobilenetV1/MobilenetV1/Conv2d_3_depthwise/BatchNorm/AssignMovingAvg?Ltrain/MobilenetV1/MobilenetV1/Conv2d_3_depthwise/BatchNorm/AssignMovingAvg_1?Jtrain/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/BatchNorm/AssignMovingAvg?Ltrain/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/BatchNorm/AssignMovingAvg_1?Jtrain/MobilenetV1/MobilenetV1/Conv2d_4_depthwise/BatchNorm/AssignMovingAvg?Ltrain/MobilenetV1/MobilenetV1/Conv2d_4_depthwise/BatchNorm/AssignMovingAvg_1?Jtrain/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/BatchNorm/AssignMovingAvg?Ltrain/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/BatchNorm/AssignMovingAvg_1?Jtrain/MobilenetV1/MobilenetV1/Conv2d_5_depthwise/BatchNorm/AssignMovingAvg?Ltrain/MobilenetV1/MobilenetV1/Conv2d_5_depthwise/BatchNorm/AssignMovingAvg_1?Jtrain/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/BatchNorm/AssignMovingAvg?Ltrain/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/BatchNorm/AssignMovingAvg_1?Jtrain/MobilenetV1/MobilenetV1/Conv2d_6_depthwise/BatchNorm/AssignMovingAvg?Ltrain/MobilenetV1/MobilenetV1/Conv2d_6_depthwise/BatchNorm/AssignMovingAvg_1?Jtrain/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/BatchNorm/AssignMovingAvg?Ltrain/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/BatchNorm/AssignMovingAvg_1?Jtrain/MobilenetV1/MobilenetV1/Conv2d_7_depthwise/BatchNorm/AssignMovingAvg?Ltrain/MobilenetV1/MobilenetV1/Conv2d_7_depthwise/BatchNorm/AssignMovingAvg_1?Jtrain/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/BatchNorm/AssignMovingAvg?Ltrain/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/BatchNorm/AssignMovingAvg_1?Jtrain/MobilenetV1/MobilenetV1/Conv2d_8_depthwise/BatchNorm/AssignMovingAvg?Ltrain/MobilenetV1/MobilenetV1/Conv2d_8_depthwise/BatchNorm/AssignMovingAvg_1?Jtrain/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/BatchNorm/AssignMovingAvg?Ltrain/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/BatchNorm/AssignMovingAvg_1?Jtrain/MobilenetV1/MobilenetV1/Conv2d_9_depthwise/BatchNorm/AssignMovingAvg?Ltrain/MobilenetV1/MobilenetV1/Conv2d_9_depthwise/BatchNorm/AssignMovingAvg_1?Jtrain/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/BatchNorm/AssignMovingAvg?Ltrain/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/BatchNorm/AssignMovingAvg_1?
Otrain/MobilenetV1/MobilenetV1/Conv2d_0/BatchNorm/AssignMovingAvg/ReadVariableOpReadVariableOpYtrain_mobilenetv1_mobilenetv1_conv2d_0_batchnorm_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02Q
Otrain/MobilenetV1/MobilenetV1/Conv2d_0/BatchNorm/AssignMovingAvg/ReadVariableOps
train/hub_input/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
train/hub_input/Mul/y?
train/hub_input/MulMulplaceholdertrain/hub_input/Mul/y:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2
train/hub_input/Muls
train/hub_input/Sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
train/hub_input/Sub/y?
train/hub_input/SubSubtrain/hub_input/Mul:z:0train/hub_input/Sub/y:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2
train/hub_input/Sub?
<train/MobilenetV1/MobilenetV1/Conv2d_0/Conv2D/ReadVariableOpReadVariableOpgtrain_mobilenetv1_mobilenetv1_conv2d_0_kernel_regularizer_l2_regularizer_l2loss_readvariableop_resource*&
_output_shapes
:*
dtype02>
<train/MobilenetV1/MobilenetV1/Conv2d_0/Conv2D/ReadVariableOp?
-train/MobilenetV1/MobilenetV1/Conv2d_0/Conv2DConv2Dtrain/hub_input/Sub:z:0Dtrain/MobilenetV1/MobilenetV1/Conv2d_0/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
2/
-train/MobilenetV1/MobilenetV1/Conv2d_0/Conv2D?
?train/MobilenetV1/MobilenetV1/Conv2d_0/BatchNorm/ReadVariableOpReadVariableOpHtrain_mobilenetv1_mobilenetv1_conv2d_0_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02A
?train/MobilenetV1/MobilenetV1/Conv2d_0/BatchNorm/ReadVariableOp?
Atrain/MobilenetV1/MobilenetV1/Conv2d_0/BatchNorm/ReadVariableOp_1ReadVariableOpJtrain_mobilenetv1_mobilenetv1_conv2d_0_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02C
Atrain/MobilenetV1/MobilenetV1/Conv2d_0/BatchNorm/ReadVariableOp_1?
Ptrain/MobilenetV1/MobilenetV1/Conv2d_0/BatchNorm/FusedBatchNormV3/ReadVariableOpReadVariableOpYtrain_mobilenetv1_mobilenetv1_conv2d_0_batchnorm_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02R
Ptrain/MobilenetV1/MobilenetV1/Conv2d_0/BatchNorm/FusedBatchNormV3/ReadVariableOp?
Rtrain/MobilenetV1/MobilenetV1/Conv2d_0/BatchNorm/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp[train_mobilenetv1_mobilenetv1_conv2d_0_batchnorm_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02T
Rtrain/MobilenetV1/MobilenetV1/Conv2d_0/BatchNorm/FusedBatchNormV3/ReadVariableOp_1?
Atrain/MobilenetV1/MobilenetV1/Conv2d_0/BatchNorm/FusedBatchNormV3FusedBatchNormV36train/MobilenetV1/MobilenetV1/Conv2d_0/Conv2D:output:0Gtrain/MobilenetV1/MobilenetV1/Conv2d_0/BatchNorm/ReadVariableOp:value:0Itrain/MobilenetV1/MobilenetV1/Conv2d_0/BatchNorm/ReadVariableOp_1:value:0Xtrain/MobilenetV1/MobilenetV1/Conv2d_0/BatchNorm/FusedBatchNormV3/ReadVariableOp:value:0Ztrain/MobilenetV1/MobilenetV1/Conv2d_0/BatchNorm/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:2C
Atrain/MobilenetV1/MobilenetV1/Conv2d_0/BatchNorm/FusedBatchNormV3?
Ftrain/MobilenetV1/MobilenetV1/Conv2d_0/BatchNorm/AssignMovingAvg/sub_1SubWtrain/MobilenetV1/MobilenetV1/Conv2d_0/BatchNorm/AssignMovingAvg/ReadVariableOp:value:0Ntrain/MobilenetV1/MobilenetV1/Conv2d_0/BatchNorm/FusedBatchNormV3:batch_mean:0*
T0*
_output_shapes
:2H
Ftrain/MobilenetV1/MobilenetV1/Conv2d_0/BatchNorm/AssignMovingAvg/sub_1?
Ftrain/MobilenetV1/MobilenetV1/Conv2d_0/BatchNorm/AssignMovingAvg/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2H
Ftrain/MobilenetV1/MobilenetV1/Conv2d_0/BatchNorm/AssignMovingAvg/sub/x?
Dtrain/MobilenetV1/MobilenetV1/Conv2d_0/BatchNorm/AssignMovingAvg/subSubOtrain/MobilenetV1/MobilenetV1/Conv2d_0/BatchNorm/AssignMovingAvg/sub/x:output:0placeholder_2*
T0*
_output_shapes
: 2F
Dtrain/MobilenetV1/MobilenetV1/Conv2d_0/BatchNorm/AssignMovingAvg/sub?
Dtrain/MobilenetV1/MobilenetV1/Conv2d_0/BatchNorm/AssignMovingAvg/mulMulJtrain/MobilenetV1/MobilenetV1/Conv2d_0/BatchNorm/AssignMovingAvg/sub_1:z:0Htrain/MobilenetV1/MobilenetV1/Conv2d_0/BatchNorm/AssignMovingAvg/sub:z:0*
T0*
_output_shapes
:2F
Dtrain/MobilenetV1/MobilenetV1/Conv2d_0/BatchNorm/AssignMovingAvg/mul?
@train/MobilenetV1/MobilenetV1/Conv2d_0/BatchNorm/AssignMovingAvgAssignSubVariableOpYtrain_mobilenetv1_mobilenetv1_conv2d_0_batchnorm_fusedbatchnormv3_readvariableop_resourceHtrain/MobilenetV1/MobilenetV1/Conv2d_0/BatchNorm/AssignMovingAvg/mul:z:0*
_output_shapes
 *
dtype02B
@train/MobilenetV1/MobilenetV1/Conv2d_0/BatchNorm/AssignMovingAvg?
Qtrain/MobilenetV1/MobilenetV1/Conv2d_0/BatchNorm/AssignMovingAvg_1/ReadVariableOpReadVariableOp[train_mobilenetv1_mobilenetv1_conv2d_0_batchnorm_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02S
Qtrain/MobilenetV1/MobilenetV1/Conv2d_0/BatchNorm/AssignMovingAvg_1/ReadVariableOp?
Htrain/MobilenetV1/MobilenetV1/Conv2d_0/BatchNorm/AssignMovingAvg_1/sub_1SubYtrain/MobilenetV1/MobilenetV1/Conv2d_0/BatchNorm/AssignMovingAvg_1/ReadVariableOp:value:0Rtrain/MobilenetV1/MobilenetV1/Conv2d_0/BatchNorm/FusedBatchNormV3:batch_variance:0*
T0*
_output_shapes
:2J
Htrain/MobilenetV1/MobilenetV1/Conv2d_0/BatchNorm/AssignMovingAvg_1/sub_1?
Htrain/MobilenetV1/MobilenetV1/Conv2d_0/BatchNorm/AssignMovingAvg_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2J
Htrain/MobilenetV1/MobilenetV1/Conv2d_0/BatchNorm/AssignMovingAvg_1/sub/x?
Ftrain/MobilenetV1/MobilenetV1/Conv2d_0/BatchNorm/AssignMovingAvg_1/subSubQtrain/MobilenetV1/MobilenetV1/Conv2d_0/BatchNorm/AssignMovingAvg_1/sub/x:output:0placeholder_2*
T0*
_output_shapes
: 2H
Ftrain/MobilenetV1/MobilenetV1/Conv2d_0/BatchNorm/AssignMovingAvg_1/sub?
Ftrain/MobilenetV1/MobilenetV1/Conv2d_0/BatchNorm/AssignMovingAvg_1/mulMulLtrain/MobilenetV1/MobilenetV1/Conv2d_0/BatchNorm/AssignMovingAvg_1/sub_1:z:0Jtrain/MobilenetV1/MobilenetV1/Conv2d_0/BatchNorm/AssignMovingAvg_1/sub:z:0*
T0*
_output_shapes
:2H
Ftrain/MobilenetV1/MobilenetV1/Conv2d_0/BatchNorm/AssignMovingAvg_1/mul?
Btrain/MobilenetV1/MobilenetV1/Conv2d_0/BatchNorm/AssignMovingAvg_1AssignSubVariableOp[train_mobilenetv1_mobilenetv1_conv2d_0_batchnorm_fusedbatchnormv3_readvariableop_1_resourceJtrain/MobilenetV1/MobilenetV1/Conv2d_0/BatchNorm/AssignMovingAvg_1/mul:z:0*
_output_shapes
 *
dtype02D
Btrain/MobilenetV1/MobilenetV1/Conv2d_0/BatchNorm/AssignMovingAvg_1?
Ytrain/MobilenetV1/MobilenetV1/Conv2d_1_depthwise/BatchNorm/AssignMovingAvg/ReadVariableOpReadVariableOpctrain_mobilenetv1_mobilenetv1_conv2d_1_depthwise_batchnorm_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02[
Ytrain/MobilenetV1/MobilenetV1/Conv2d_1_depthwise/BatchNorm/AssignMovingAvg/ReadVariableOp?
,train/MobilenetV1/MobilenetV1/Conv2d_0/Relu6Relu6Etrain/MobilenetV1/MobilenetV1/Conv2d_0/BatchNorm/FusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+???????????????????????????2.
,train/MobilenetV1/MobilenetV1/Conv2d_0/Relu6?
Itrain/MobilenetV1/MobilenetV1/Conv2d_1_depthwise/depthwise/ReadVariableOpReadVariableOpRtrain_mobilenetv1_mobilenetv1_conv2d_1_depthwise_depthwise_readvariableop_resource*&
_output_shapes
:*
dtype02K
Itrain/MobilenetV1/MobilenetV1/Conv2d_1_depthwise/depthwise/ReadVariableOp?
:train/MobilenetV1/MobilenetV1/Conv2d_1_depthwise/depthwiseDepthwiseConv2dNative:train/MobilenetV1/MobilenetV1/Conv2d_0/Relu6:activations:0Qtrain/MobilenetV1/MobilenetV1/Conv2d_1_depthwise/depthwise/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
2<
:train/MobilenetV1/MobilenetV1/Conv2d_1_depthwise/depthwise?
Itrain/MobilenetV1/MobilenetV1/Conv2d_1_depthwise/BatchNorm/ReadVariableOpReadVariableOpRtrain_mobilenetv1_mobilenetv1_conv2d_1_depthwise_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02K
Itrain/MobilenetV1/MobilenetV1/Conv2d_1_depthwise/BatchNorm/ReadVariableOp?
Ktrain/MobilenetV1/MobilenetV1/Conv2d_1_depthwise/BatchNorm/ReadVariableOp_1ReadVariableOpTtrain_mobilenetv1_mobilenetv1_conv2d_1_depthwise_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02M
Ktrain/MobilenetV1/MobilenetV1/Conv2d_1_depthwise/BatchNorm/ReadVariableOp_1?
Ztrain/MobilenetV1/MobilenetV1/Conv2d_1_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOpReadVariableOpctrain_mobilenetv1_mobilenetv1_conv2d_1_depthwise_batchnorm_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02\
Ztrain/MobilenetV1/MobilenetV1/Conv2d_1_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp?
\train/MobilenetV1/MobilenetV1/Conv2d_1_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpetrain_mobilenetv1_mobilenetv1_conv2d_1_depthwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02^
\train/MobilenetV1/MobilenetV1/Conv2d_1_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1?
Ktrain/MobilenetV1/MobilenetV1/Conv2d_1_depthwise/BatchNorm/FusedBatchNormV3FusedBatchNormV3Ctrain/MobilenetV1/MobilenetV1/Conv2d_1_depthwise/depthwise:output:0Qtrain/MobilenetV1/MobilenetV1/Conv2d_1_depthwise/BatchNorm/ReadVariableOp:value:0Strain/MobilenetV1/MobilenetV1/Conv2d_1_depthwise/BatchNorm/ReadVariableOp_1:value:0btrain/MobilenetV1/MobilenetV1/Conv2d_1_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp:value:0dtrain/MobilenetV1/MobilenetV1/Conv2d_1_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:2M
Ktrain/MobilenetV1/MobilenetV1/Conv2d_1_depthwise/BatchNorm/FusedBatchNormV3?
Ptrain/MobilenetV1/MobilenetV1/Conv2d_1_depthwise/BatchNorm/AssignMovingAvg/sub_1Subatrain/MobilenetV1/MobilenetV1/Conv2d_1_depthwise/BatchNorm/AssignMovingAvg/ReadVariableOp:value:0Xtrain/MobilenetV1/MobilenetV1/Conv2d_1_depthwise/BatchNorm/FusedBatchNormV3:batch_mean:0*
T0*
_output_shapes
:2R
Ptrain/MobilenetV1/MobilenetV1/Conv2d_1_depthwise/BatchNorm/AssignMovingAvg/sub_1?
Ptrain/MobilenetV1/MobilenetV1/Conv2d_1_depthwise/BatchNorm/AssignMovingAvg/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2R
Ptrain/MobilenetV1/MobilenetV1/Conv2d_1_depthwise/BatchNorm/AssignMovingAvg/sub/x?
Ntrain/MobilenetV1/MobilenetV1/Conv2d_1_depthwise/BatchNorm/AssignMovingAvg/subSubYtrain/MobilenetV1/MobilenetV1/Conv2d_1_depthwise/BatchNorm/AssignMovingAvg/sub/x:output:0placeholder_2*
T0*
_output_shapes
: 2P
Ntrain/MobilenetV1/MobilenetV1/Conv2d_1_depthwise/BatchNorm/AssignMovingAvg/sub?
Ntrain/MobilenetV1/MobilenetV1/Conv2d_1_depthwise/BatchNorm/AssignMovingAvg/mulMulTtrain/MobilenetV1/MobilenetV1/Conv2d_1_depthwise/BatchNorm/AssignMovingAvg/sub_1:z:0Rtrain/MobilenetV1/MobilenetV1/Conv2d_1_depthwise/BatchNorm/AssignMovingAvg/sub:z:0*
T0*
_output_shapes
:2P
Ntrain/MobilenetV1/MobilenetV1/Conv2d_1_depthwise/BatchNorm/AssignMovingAvg/mul?
Jtrain/MobilenetV1/MobilenetV1/Conv2d_1_depthwise/BatchNorm/AssignMovingAvgAssignSubVariableOpctrain_mobilenetv1_mobilenetv1_conv2d_1_depthwise_batchnorm_fusedbatchnormv3_readvariableop_resourceRtrain/MobilenetV1/MobilenetV1/Conv2d_1_depthwise/BatchNorm/AssignMovingAvg/mul:z:0*
_output_shapes
 *
dtype02L
Jtrain/MobilenetV1/MobilenetV1/Conv2d_1_depthwise/BatchNorm/AssignMovingAvg?
[train/MobilenetV1/MobilenetV1/Conv2d_1_depthwise/BatchNorm/AssignMovingAvg_1/ReadVariableOpReadVariableOpetrain_mobilenetv1_mobilenetv1_conv2d_1_depthwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02]
[train/MobilenetV1/MobilenetV1/Conv2d_1_depthwise/BatchNorm/AssignMovingAvg_1/ReadVariableOp?
Rtrain/MobilenetV1/MobilenetV1/Conv2d_1_depthwise/BatchNorm/AssignMovingAvg_1/sub_1Subctrain/MobilenetV1/MobilenetV1/Conv2d_1_depthwise/BatchNorm/AssignMovingAvg_1/ReadVariableOp:value:0\train/MobilenetV1/MobilenetV1/Conv2d_1_depthwise/BatchNorm/FusedBatchNormV3:batch_variance:0*
T0*
_output_shapes
:2T
Rtrain/MobilenetV1/MobilenetV1/Conv2d_1_depthwise/BatchNorm/AssignMovingAvg_1/sub_1?
Rtrain/MobilenetV1/MobilenetV1/Conv2d_1_depthwise/BatchNorm/AssignMovingAvg_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2T
Rtrain/MobilenetV1/MobilenetV1/Conv2d_1_depthwise/BatchNorm/AssignMovingAvg_1/sub/x?
Ptrain/MobilenetV1/MobilenetV1/Conv2d_1_depthwise/BatchNorm/AssignMovingAvg_1/subSub[train/MobilenetV1/MobilenetV1/Conv2d_1_depthwise/BatchNorm/AssignMovingAvg_1/sub/x:output:0placeholder_2*
T0*
_output_shapes
: 2R
Ptrain/MobilenetV1/MobilenetV1/Conv2d_1_depthwise/BatchNorm/AssignMovingAvg_1/sub?
Ptrain/MobilenetV1/MobilenetV1/Conv2d_1_depthwise/BatchNorm/AssignMovingAvg_1/mulMulVtrain/MobilenetV1/MobilenetV1/Conv2d_1_depthwise/BatchNorm/AssignMovingAvg_1/sub_1:z:0Ttrain/MobilenetV1/MobilenetV1/Conv2d_1_depthwise/BatchNorm/AssignMovingAvg_1/sub:z:0*
T0*
_output_shapes
:2R
Ptrain/MobilenetV1/MobilenetV1/Conv2d_1_depthwise/BatchNorm/AssignMovingAvg_1/mul?
Ltrain/MobilenetV1/MobilenetV1/Conv2d_1_depthwise/BatchNorm/AssignMovingAvg_1AssignSubVariableOpetrain_mobilenetv1_mobilenetv1_conv2d_1_depthwise_batchnorm_fusedbatchnormv3_readvariableop_1_resourceTtrain/MobilenetV1/MobilenetV1/Conv2d_1_depthwise/BatchNorm/AssignMovingAvg_1/mul:z:0*
_output_shapes
 *
dtype02N
Ltrain/MobilenetV1/MobilenetV1/Conv2d_1_depthwise/BatchNorm/AssignMovingAvg_1?
Ytrain/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/BatchNorm/AssignMovingAvg/ReadVariableOpReadVariableOpctrain_mobilenetv1_mobilenetv1_conv2d_1_pointwise_batchnorm_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02[
Ytrain/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/BatchNorm/AssignMovingAvg/ReadVariableOp?
6train/MobilenetV1/MobilenetV1/Conv2d_1_depthwise/Relu6Relu6Otrain/MobilenetV1/MobilenetV1/Conv2d_1_depthwise/BatchNorm/FusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+???????????????????????????28
6train/MobilenetV1/MobilenetV1/Conv2d_1_depthwise/Relu6?
Ftrain/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/Conv2D/ReadVariableOpReadVariableOpqtrain_mobilenetv1_mobilenetv1_conv2d_1_pointwise_kernel_regularizer_l2_regularizer_l2loss_readvariableop_resource*&
_output_shapes
:0*
dtype02H
Ftrain/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/Conv2D/ReadVariableOp?
7train/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/Conv2DConv2DDtrain/MobilenetV1/MobilenetV1/Conv2d_1_depthwise/Relu6:activations:0Ntrain/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????0*
paddingSAME*
strides
29
7train/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/Conv2D?
Itrain/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/BatchNorm/ReadVariableOpReadVariableOpRtrain_mobilenetv1_mobilenetv1_conv2d_1_pointwise_batchnorm_readvariableop_resource*
_output_shapes
:0*
dtype02K
Itrain/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/BatchNorm/ReadVariableOp?
Ktrain/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/BatchNorm/ReadVariableOp_1ReadVariableOpTtrain_mobilenetv1_mobilenetv1_conv2d_1_pointwise_batchnorm_readvariableop_1_resource*
_output_shapes
:0*
dtype02M
Ktrain/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/BatchNorm/ReadVariableOp_1?
Ztrain/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOpReadVariableOpctrain_mobilenetv1_mobilenetv1_conv2d_1_pointwise_batchnorm_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02\
Ztrain/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp?
\train/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpetrain_mobilenetv1_mobilenetv1_conv2d_1_pointwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02^
\train/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1?
Ktrain/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/BatchNorm/FusedBatchNormV3FusedBatchNormV3@train/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/Conv2D:output:0Qtrain/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/BatchNorm/ReadVariableOp:value:0Strain/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/BatchNorm/ReadVariableOp_1:value:0btrain/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp:value:0dtrain/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????0:0:0:0:0:*
epsilon%o?:2M
Ktrain/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/BatchNorm/FusedBatchNormV3?
Ptrain/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/BatchNorm/AssignMovingAvg/sub_1Subatrain/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/BatchNorm/AssignMovingAvg/ReadVariableOp:value:0Xtrain/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/BatchNorm/FusedBatchNormV3:batch_mean:0*
T0*
_output_shapes
:02R
Ptrain/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/BatchNorm/AssignMovingAvg/sub_1?
Ptrain/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/BatchNorm/AssignMovingAvg/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2R
Ptrain/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/BatchNorm/AssignMovingAvg/sub/x?
Ntrain/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/BatchNorm/AssignMovingAvg/subSubYtrain/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/BatchNorm/AssignMovingAvg/sub/x:output:0placeholder_2*
T0*
_output_shapes
: 2P
Ntrain/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/BatchNorm/AssignMovingAvg/sub?
Ntrain/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/BatchNorm/AssignMovingAvg/mulMulTtrain/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/BatchNorm/AssignMovingAvg/sub_1:z:0Rtrain/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/BatchNorm/AssignMovingAvg/sub:z:0*
T0*
_output_shapes
:02P
Ntrain/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/BatchNorm/AssignMovingAvg/mul?
Jtrain/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/BatchNorm/AssignMovingAvgAssignSubVariableOpctrain_mobilenetv1_mobilenetv1_conv2d_1_pointwise_batchnorm_fusedbatchnormv3_readvariableop_resourceRtrain/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/BatchNorm/AssignMovingAvg/mul:z:0*
_output_shapes
 *
dtype02L
Jtrain/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/BatchNorm/AssignMovingAvg?
[train/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/BatchNorm/AssignMovingAvg_1/ReadVariableOpReadVariableOpetrain_mobilenetv1_mobilenetv1_conv2d_1_pointwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02]
[train/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/BatchNorm/AssignMovingAvg_1/ReadVariableOp?
Rtrain/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/BatchNorm/AssignMovingAvg_1/sub_1Subctrain/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/BatchNorm/AssignMovingAvg_1/ReadVariableOp:value:0\train/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/BatchNorm/FusedBatchNormV3:batch_variance:0*
T0*
_output_shapes
:02T
Rtrain/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/BatchNorm/AssignMovingAvg_1/sub_1?
Rtrain/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/BatchNorm/AssignMovingAvg_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2T
Rtrain/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/BatchNorm/AssignMovingAvg_1/sub/x?
Ptrain/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/BatchNorm/AssignMovingAvg_1/subSub[train/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/BatchNorm/AssignMovingAvg_1/sub/x:output:0placeholder_2*
T0*
_output_shapes
: 2R
Ptrain/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/BatchNorm/AssignMovingAvg_1/sub?
Ptrain/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/BatchNorm/AssignMovingAvg_1/mulMulVtrain/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/BatchNorm/AssignMovingAvg_1/sub_1:z:0Ttrain/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/BatchNorm/AssignMovingAvg_1/sub:z:0*
T0*
_output_shapes
:02R
Ptrain/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/BatchNorm/AssignMovingAvg_1/mul?
Ltrain/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/BatchNorm/AssignMovingAvg_1AssignSubVariableOpetrain_mobilenetv1_mobilenetv1_conv2d_1_pointwise_batchnorm_fusedbatchnormv3_readvariableop_1_resourceTtrain/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/BatchNorm/AssignMovingAvg_1/mul:z:0*
_output_shapes
 *
dtype02N
Ltrain/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/BatchNorm/AssignMovingAvg_1?
Ytrain/MobilenetV1/MobilenetV1/Conv2d_2_depthwise/BatchNorm/AssignMovingAvg/ReadVariableOpReadVariableOpctrain_mobilenetv1_mobilenetv1_conv2d_2_depthwise_batchnorm_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02[
Ytrain/MobilenetV1/MobilenetV1/Conv2d_2_depthwise/BatchNorm/AssignMovingAvg/ReadVariableOp?
6train/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/Relu6Relu6Otrain/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/BatchNorm/FusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+???????????????????????????028
6train/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/Relu6?
Itrain/MobilenetV1/MobilenetV1/Conv2d_2_depthwise/depthwise/ReadVariableOpReadVariableOpRtrain_mobilenetv1_mobilenetv1_conv2d_2_depthwise_depthwise_readvariableop_resource*&
_output_shapes
:0*
dtype02K
Itrain/MobilenetV1/MobilenetV1/Conv2d_2_depthwise/depthwise/ReadVariableOp?
:train/MobilenetV1/MobilenetV1/Conv2d_2_depthwise/depthwiseDepthwiseConv2dNativeDtrain/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/Relu6:activations:0Qtrain/MobilenetV1/MobilenetV1/Conv2d_2_depthwise/depthwise/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????0*
paddingSAME*
strides
2<
:train/MobilenetV1/MobilenetV1/Conv2d_2_depthwise/depthwise?
Itrain/MobilenetV1/MobilenetV1/Conv2d_2_depthwise/BatchNorm/ReadVariableOpReadVariableOpRtrain_mobilenetv1_mobilenetv1_conv2d_2_depthwise_batchnorm_readvariableop_resource*
_output_shapes
:0*
dtype02K
Itrain/MobilenetV1/MobilenetV1/Conv2d_2_depthwise/BatchNorm/ReadVariableOp?
Ktrain/MobilenetV1/MobilenetV1/Conv2d_2_depthwise/BatchNorm/ReadVariableOp_1ReadVariableOpTtrain_mobilenetv1_mobilenetv1_conv2d_2_depthwise_batchnorm_readvariableop_1_resource*
_output_shapes
:0*
dtype02M
Ktrain/MobilenetV1/MobilenetV1/Conv2d_2_depthwise/BatchNorm/ReadVariableOp_1?
Ztrain/MobilenetV1/MobilenetV1/Conv2d_2_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOpReadVariableOpctrain_mobilenetv1_mobilenetv1_conv2d_2_depthwise_batchnorm_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02\
Ztrain/MobilenetV1/MobilenetV1/Conv2d_2_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp?
\train/MobilenetV1/MobilenetV1/Conv2d_2_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpetrain_mobilenetv1_mobilenetv1_conv2d_2_depthwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02^
\train/MobilenetV1/MobilenetV1/Conv2d_2_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1?
Ktrain/MobilenetV1/MobilenetV1/Conv2d_2_depthwise/BatchNorm/FusedBatchNormV3FusedBatchNormV3Ctrain/MobilenetV1/MobilenetV1/Conv2d_2_depthwise/depthwise:output:0Qtrain/MobilenetV1/MobilenetV1/Conv2d_2_depthwise/BatchNorm/ReadVariableOp:value:0Strain/MobilenetV1/MobilenetV1/Conv2d_2_depthwise/BatchNorm/ReadVariableOp_1:value:0btrain/MobilenetV1/MobilenetV1/Conv2d_2_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp:value:0dtrain/MobilenetV1/MobilenetV1/Conv2d_2_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????0:0:0:0:0:*
epsilon%o?:2M
Ktrain/MobilenetV1/MobilenetV1/Conv2d_2_depthwise/BatchNorm/FusedBatchNormV3?
Ptrain/MobilenetV1/MobilenetV1/Conv2d_2_depthwise/BatchNorm/AssignMovingAvg/sub_1Subatrain/MobilenetV1/MobilenetV1/Conv2d_2_depthwise/BatchNorm/AssignMovingAvg/ReadVariableOp:value:0Xtrain/MobilenetV1/MobilenetV1/Conv2d_2_depthwise/BatchNorm/FusedBatchNormV3:batch_mean:0*
T0*
_output_shapes
:02R
Ptrain/MobilenetV1/MobilenetV1/Conv2d_2_depthwise/BatchNorm/AssignMovingAvg/sub_1?
Ptrain/MobilenetV1/MobilenetV1/Conv2d_2_depthwise/BatchNorm/AssignMovingAvg/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2R
Ptrain/MobilenetV1/MobilenetV1/Conv2d_2_depthwise/BatchNorm/AssignMovingAvg/sub/x?
Ntrain/MobilenetV1/MobilenetV1/Conv2d_2_depthwise/BatchNorm/AssignMovingAvg/subSubYtrain/MobilenetV1/MobilenetV1/Conv2d_2_depthwise/BatchNorm/AssignMovingAvg/sub/x:output:0placeholder_2*
T0*
_output_shapes
: 2P
Ntrain/MobilenetV1/MobilenetV1/Conv2d_2_depthwise/BatchNorm/AssignMovingAvg/sub?
Ntrain/MobilenetV1/MobilenetV1/Conv2d_2_depthwise/BatchNorm/AssignMovingAvg/mulMulTtrain/MobilenetV1/MobilenetV1/Conv2d_2_depthwise/BatchNorm/AssignMovingAvg/sub_1:z:0Rtrain/MobilenetV1/MobilenetV1/Conv2d_2_depthwise/BatchNorm/AssignMovingAvg/sub:z:0*
T0*
_output_shapes
:02P
Ntrain/MobilenetV1/MobilenetV1/Conv2d_2_depthwise/BatchNorm/AssignMovingAvg/mul?
Jtrain/MobilenetV1/MobilenetV1/Conv2d_2_depthwise/BatchNorm/AssignMovingAvgAssignSubVariableOpctrain_mobilenetv1_mobilenetv1_conv2d_2_depthwise_batchnorm_fusedbatchnormv3_readvariableop_resourceRtrain/MobilenetV1/MobilenetV1/Conv2d_2_depthwise/BatchNorm/AssignMovingAvg/mul:z:0*
_output_shapes
 *
dtype02L
Jtrain/MobilenetV1/MobilenetV1/Conv2d_2_depthwise/BatchNorm/AssignMovingAvg?
[train/MobilenetV1/MobilenetV1/Conv2d_2_depthwise/BatchNorm/AssignMovingAvg_1/ReadVariableOpReadVariableOpetrain_mobilenetv1_mobilenetv1_conv2d_2_depthwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02]
[train/MobilenetV1/MobilenetV1/Conv2d_2_depthwise/BatchNorm/AssignMovingAvg_1/ReadVariableOp?
Rtrain/MobilenetV1/MobilenetV1/Conv2d_2_depthwise/BatchNorm/AssignMovingAvg_1/sub_1Subctrain/MobilenetV1/MobilenetV1/Conv2d_2_depthwise/BatchNorm/AssignMovingAvg_1/ReadVariableOp:value:0\train/MobilenetV1/MobilenetV1/Conv2d_2_depthwise/BatchNorm/FusedBatchNormV3:batch_variance:0*
T0*
_output_shapes
:02T
Rtrain/MobilenetV1/MobilenetV1/Conv2d_2_depthwise/BatchNorm/AssignMovingAvg_1/sub_1?
Rtrain/MobilenetV1/MobilenetV1/Conv2d_2_depthwise/BatchNorm/AssignMovingAvg_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2T
Rtrain/MobilenetV1/MobilenetV1/Conv2d_2_depthwise/BatchNorm/AssignMovingAvg_1/sub/x?
Ptrain/MobilenetV1/MobilenetV1/Conv2d_2_depthwise/BatchNorm/AssignMovingAvg_1/subSub[train/MobilenetV1/MobilenetV1/Conv2d_2_depthwise/BatchNorm/AssignMovingAvg_1/sub/x:output:0placeholder_2*
T0*
_output_shapes
: 2R
Ptrain/MobilenetV1/MobilenetV1/Conv2d_2_depthwise/BatchNorm/AssignMovingAvg_1/sub?
Ptrain/MobilenetV1/MobilenetV1/Conv2d_2_depthwise/BatchNorm/AssignMovingAvg_1/mulMulVtrain/MobilenetV1/MobilenetV1/Conv2d_2_depthwise/BatchNorm/AssignMovingAvg_1/sub_1:z:0Ttrain/MobilenetV1/MobilenetV1/Conv2d_2_depthwise/BatchNorm/AssignMovingAvg_1/sub:z:0*
T0*
_output_shapes
:02R
Ptrain/MobilenetV1/MobilenetV1/Conv2d_2_depthwise/BatchNorm/AssignMovingAvg_1/mul?
Ltrain/MobilenetV1/MobilenetV1/Conv2d_2_depthwise/BatchNorm/AssignMovingAvg_1AssignSubVariableOpetrain_mobilenetv1_mobilenetv1_conv2d_2_depthwise_batchnorm_fusedbatchnormv3_readvariableop_1_resourceTtrain/MobilenetV1/MobilenetV1/Conv2d_2_depthwise/BatchNorm/AssignMovingAvg_1/mul:z:0*
_output_shapes
 *
dtype02N
Ltrain/MobilenetV1/MobilenetV1/Conv2d_2_depthwise/BatchNorm/AssignMovingAvg_1?
Ytrain/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/BatchNorm/AssignMovingAvg/ReadVariableOpReadVariableOpctrain_mobilenetv1_mobilenetv1_conv2d_2_pointwise_batchnorm_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:`*
dtype02[
Ytrain/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/BatchNorm/AssignMovingAvg/ReadVariableOp?
6train/MobilenetV1/MobilenetV1/Conv2d_2_depthwise/Relu6Relu6Otrain/MobilenetV1/MobilenetV1/Conv2d_2_depthwise/BatchNorm/FusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+???????????????????????????028
6train/MobilenetV1/MobilenetV1/Conv2d_2_depthwise/Relu6?
Ftrain/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/Conv2D/ReadVariableOpReadVariableOpqtrain_mobilenetv1_mobilenetv1_conv2d_2_pointwise_kernel_regularizer_l2_regularizer_l2loss_readvariableop_resource*&
_output_shapes
:0`*
dtype02H
Ftrain/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/Conv2D/ReadVariableOp?
7train/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/Conv2DConv2DDtrain/MobilenetV1/MobilenetV1/Conv2d_2_depthwise/Relu6:activations:0Ntrain/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????`*
paddingSAME*
strides
29
7train/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/Conv2D?
Itrain/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/BatchNorm/ReadVariableOpReadVariableOpRtrain_mobilenetv1_mobilenetv1_conv2d_2_pointwise_batchnorm_readvariableop_resource*
_output_shapes
:`*
dtype02K
Itrain/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/BatchNorm/ReadVariableOp?
Ktrain/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/BatchNorm/ReadVariableOp_1ReadVariableOpTtrain_mobilenetv1_mobilenetv1_conv2d_2_pointwise_batchnorm_readvariableop_1_resource*
_output_shapes
:`*
dtype02M
Ktrain/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/BatchNorm/ReadVariableOp_1?
Ztrain/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOpReadVariableOpctrain_mobilenetv1_mobilenetv1_conv2d_2_pointwise_batchnorm_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:`*
dtype02\
Ztrain/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp?
\train/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpetrain_mobilenetv1_mobilenetv1_conv2d_2_pointwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:`*
dtype02^
\train/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1?
Ktrain/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/BatchNorm/FusedBatchNormV3FusedBatchNormV3@train/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/Conv2D:output:0Qtrain/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/BatchNorm/ReadVariableOp:value:0Strain/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/BatchNorm/ReadVariableOp_1:value:0btrain/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp:value:0dtrain/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????`:`:`:`:`:*
epsilon%o?:2M
Ktrain/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/BatchNorm/FusedBatchNormV3?
Ptrain/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/BatchNorm/AssignMovingAvg/sub_1Subatrain/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/BatchNorm/AssignMovingAvg/ReadVariableOp:value:0Xtrain/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/BatchNorm/FusedBatchNormV3:batch_mean:0*
T0*
_output_shapes
:`2R
Ptrain/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/BatchNorm/AssignMovingAvg/sub_1?
Ptrain/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/BatchNorm/AssignMovingAvg/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2R
Ptrain/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/BatchNorm/AssignMovingAvg/sub/x?
Ntrain/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/BatchNorm/AssignMovingAvg/subSubYtrain/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/BatchNorm/AssignMovingAvg/sub/x:output:0placeholder_2*
T0*
_output_shapes
: 2P
Ntrain/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/BatchNorm/AssignMovingAvg/sub?
Ntrain/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/BatchNorm/AssignMovingAvg/mulMulTtrain/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/BatchNorm/AssignMovingAvg/sub_1:z:0Rtrain/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/BatchNorm/AssignMovingAvg/sub:z:0*
T0*
_output_shapes
:`2P
Ntrain/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/BatchNorm/AssignMovingAvg/mul?
Jtrain/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/BatchNorm/AssignMovingAvgAssignSubVariableOpctrain_mobilenetv1_mobilenetv1_conv2d_2_pointwise_batchnorm_fusedbatchnormv3_readvariableop_resourceRtrain/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/BatchNorm/AssignMovingAvg/mul:z:0*
_output_shapes
 *
dtype02L
Jtrain/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/BatchNorm/AssignMovingAvg?
[train/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/BatchNorm/AssignMovingAvg_1/ReadVariableOpReadVariableOpetrain_mobilenetv1_mobilenetv1_conv2d_2_pointwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:`*
dtype02]
[train/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/BatchNorm/AssignMovingAvg_1/ReadVariableOp?
Rtrain/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/BatchNorm/AssignMovingAvg_1/sub_1Subctrain/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/BatchNorm/AssignMovingAvg_1/ReadVariableOp:value:0\train/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/BatchNorm/FusedBatchNormV3:batch_variance:0*
T0*
_output_shapes
:`2T
Rtrain/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/BatchNorm/AssignMovingAvg_1/sub_1?
Rtrain/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/BatchNorm/AssignMovingAvg_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2T
Rtrain/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/BatchNorm/AssignMovingAvg_1/sub/x?
Ptrain/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/BatchNorm/AssignMovingAvg_1/subSub[train/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/BatchNorm/AssignMovingAvg_1/sub/x:output:0placeholder_2*
T0*
_output_shapes
: 2R
Ptrain/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/BatchNorm/AssignMovingAvg_1/sub?
Ptrain/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/BatchNorm/AssignMovingAvg_1/mulMulVtrain/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/BatchNorm/AssignMovingAvg_1/sub_1:z:0Ttrain/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/BatchNorm/AssignMovingAvg_1/sub:z:0*
T0*
_output_shapes
:`2R
Ptrain/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/BatchNorm/AssignMovingAvg_1/mul?
Ltrain/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/BatchNorm/AssignMovingAvg_1AssignSubVariableOpetrain_mobilenetv1_mobilenetv1_conv2d_2_pointwise_batchnorm_fusedbatchnormv3_readvariableop_1_resourceTtrain/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/BatchNorm/AssignMovingAvg_1/mul:z:0*
_output_shapes
 *
dtype02N
Ltrain/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/BatchNorm/AssignMovingAvg_1?
Ytrain/MobilenetV1/MobilenetV1/Conv2d_3_depthwise/BatchNorm/AssignMovingAvg/ReadVariableOpReadVariableOpctrain_mobilenetv1_mobilenetv1_conv2d_3_depthwise_batchnorm_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:`*
dtype02[
Ytrain/MobilenetV1/MobilenetV1/Conv2d_3_depthwise/BatchNorm/AssignMovingAvg/ReadVariableOp?
6train/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/Relu6Relu6Otrain/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/BatchNorm/FusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+???????????????????????????`28
6train/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/Relu6?
Itrain/MobilenetV1/MobilenetV1/Conv2d_3_depthwise/depthwise/ReadVariableOpReadVariableOpRtrain_mobilenetv1_mobilenetv1_conv2d_3_depthwise_depthwise_readvariableop_resource*&
_output_shapes
:`*
dtype02K
Itrain/MobilenetV1/MobilenetV1/Conv2d_3_depthwise/depthwise/ReadVariableOp?
:train/MobilenetV1/MobilenetV1/Conv2d_3_depthwise/depthwiseDepthwiseConv2dNativeDtrain/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/Relu6:activations:0Qtrain/MobilenetV1/MobilenetV1/Conv2d_3_depthwise/depthwise/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????`*
paddingSAME*
strides
2<
:train/MobilenetV1/MobilenetV1/Conv2d_3_depthwise/depthwise?
Itrain/MobilenetV1/MobilenetV1/Conv2d_3_depthwise/BatchNorm/ReadVariableOpReadVariableOpRtrain_mobilenetv1_mobilenetv1_conv2d_3_depthwise_batchnorm_readvariableop_resource*
_output_shapes
:`*
dtype02K
Itrain/MobilenetV1/MobilenetV1/Conv2d_3_depthwise/BatchNorm/ReadVariableOp?
Ktrain/MobilenetV1/MobilenetV1/Conv2d_3_depthwise/BatchNorm/ReadVariableOp_1ReadVariableOpTtrain_mobilenetv1_mobilenetv1_conv2d_3_depthwise_batchnorm_readvariableop_1_resource*
_output_shapes
:`*
dtype02M
Ktrain/MobilenetV1/MobilenetV1/Conv2d_3_depthwise/BatchNorm/ReadVariableOp_1?
Ztrain/MobilenetV1/MobilenetV1/Conv2d_3_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOpReadVariableOpctrain_mobilenetv1_mobilenetv1_conv2d_3_depthwise_batchnorm_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:`*
dtype02\
Ztrain/MobilenetV1/MobilenetV1/Conv2d_3_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp?
\train/MobilenetV1/MobilenetV1/Conv2d_3_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpetrain_mobilenetv1_mobilenetv1_conv2d_3_depthwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:`*
dtype02^
\train/MobilenetV1/MobilenetV1/Conv2d_3_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1?
Ktrain/MobilenetV1/MobilenetV1/Conv2d_3_depthwise/BatchNorm/FusedBatchNormV3FusedBatchNormV3Ctrain/MobilenetV1/MobilenetV1/Conv2d_3_depthwise/depthwise:output:0Qtrain/MobilenetV1/MobilenetV1/Conv2d_3_depthwise/BatchNorm/ReadVariableOp:value:0Strain/MobilenetV1/MobilenetV1/Conv2d_3_depthwise/BatchNorm/ReadVariableOp_1:value:0btrain/MobilenetV1/MobilenetV1/Conv2d_3_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp:value:0dtrain/MobilenetV1/MobilenetV1/Conv2d_3_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????`:`:`:`:`:*
epsilon%o?:2M
Ktrain/MobilenetV1/MobilenetV1/Conv2d_3_depthwise/BatchNorm/FusedBatchNormV3?
Ptrain/MobilenetV1/MobilenetV1/Conv2d_3_depthwise/BatchNorm/AssignMovingAvg/sub_1Subatrain/MobilenetV1/MobilenetV1/Conv2d_3_depthwise/BatchNorm/AssignMovingAvg/ReadVariableOp:value:0Xtrain/MobilenetV1/MobilenetV1/Conv2d_3_depthwise/BatchNorm/FusedBatchNormV3:batch_mean:0*
T0*
_output_shapes
:`2R
Ptrain/MobilenetV1/MobilenetV1/Conv2d_3_depthwise/BatchNorm/AssignMovingAvg/sub_1?
Ptrain/MobilenetV1/MobilenetV1/Conv2d_3_depthwise/BatchNorm/AssignMovingAvg/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2R
Ptrain/MobilenetV1/MobilenetV1/Conv2d_3_depthwise/BatchNorm/AssignMovingAvg/sub/x?
Ntrain/MobilenetV1/MobilenetV1/Conv2d_3_depthwise/BatchNorm/AssignMovingAvg/subSubYtrain/MobilenetV1/MobilenetV1/Conv2d_3_depthwise/BatchNorm/AssignMovingAvg/sub/x:output:0placeholder_2*
T0*
_output_shapes
: 2P
Ntrain/MobilenetV1/MobilenetV1/Conv2d_3_depthwise/BatchNorm/AssignMovingAvg/sub?
Ntrain/MobilenetV1/MobilenetV1/Conv2d_3_depthwise/BatchNorm/AssignMovingAvg/mulMulTtrain/MobilenetV1/MobilenetV1/Conv2d_3_depthwise/BatchNorm/AssignMovingAvg/sub_1:z:0Rtrain/MobilenetV1/MobilenetV1/Conv2d_3_depthwise/BatchNorm/AssignMovingAvg/sub:z:0*
T0*
_output_shapes
:`2P
Ntrain/MobilenetV1/MobilenetV1/Conv2d_3_depthwise/BatchNorm/AssignMovingAvg/mul?
Jtrain/MobilenetV1/MobilenetV1/Conv2d_3_depthwise/BatchNorm/AssignMovingAvgAssignSubVariableOpctrain_mobilenetv1_mobilenetv1_conv2d_3_depthwise_batchnorm_fusedbatchnormv3_readvariableop_resourceRtrain/MobilenetV1/MobilenetV1/Conv2d_3_depthwise/BatchNorm/AssignMovingAvg/mul:z:0*
_output_shapes
 *
dtype02L
Jtrain/MobilenetV1/MobilenetV1/Conv2d_3_depthwise/BatchNorm/AssignMovingAvg?
[train/MobilenetV1/MobilenetV1/Conv2d_3_depthwise/BatchNorm/AssignMovingAvg_1/ReadVariableOpReadVariableOpetrain_mobilenetv1_mobilenetv1_conv2d_3_depthwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:`*
dtype02]
[train/MobilenetV1/MobilenetV1/Conv2d_3_depthwise/BatchNorm/AssignMovingAvg_1/ReadVariableOp?
Rtrain/MobilenetV1/MobilenetV1/Conv2d_3_depthwise/BatchNorm/AssignMovingAvg_1/sub_1Subctrain/MobilenetV1/MobilenetV1/Conv2d_3_depthwise/BatchNorm/AssignMovingAvg_1/ReadVariableOp:value:0\train/MobilenetV1/MobilenetV1/Conv2d_3_depthwise/BatchNorm/FusedBatchNormV3:batch_variance:0*
T0*
_output_shapes
:`2T
Rtrain/MobilenetV1/MobilenetV1/Conv2d_3_depthwise/BatchNorm/AssignMovingAvg_1/sub_1?
Rtrain/MobilenetV1/MobilenetV1/Conv2d_3_depthwise/BatchNorm/AssignMovingAvg_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2T
Rtrain/MobilenetV1/MobilenetV1/Conv2d_3_depthwise/BatchNorm/AssignMovingAvg_1/sub/x?
Ptrain/MobilenetV1/MobilenetV1/Conv2d_3_depthwise/BatchNorm/AssignMovingAvg_1/subSub[train/MobilenetV1/MobilenetV1/Conv2d_3_depthwise/BatchNorm/AssignMovingAvg_1/sub/x:output:0placeholder_2*
T0*
_output_shapes
: 2R
Ptrain/MobilenetV1/MobilenetV1/Conv2d_3_depthwise/BatchNorm/AssignMovingAvg_1/sub?
Ptrain/MobilenetV1/MobilenetV1/Conv2d_3_depthwise/BatchNorm/AssignMovingAvg_1/mulMulVtrain/MobilenetV1/MobilenetV1/Conv2d_3_depthwise/BatchNorm/AssignMovingAvg_1/sub_1:z:0Ttrain/MobilenetV1/MobilenetV1/Conv2d_3_depthwise/BatchNorm/AssignMovingAvg_1/sub:z:0*
T0*
_output_shapes
:`2R
Ptrain/MobilenetV1/MobilenetV1/Conv2d_3_depthwise/BatchNorm/AssignMovingAvg_1/mul?
Ltrain/MobilenetV1/MobilenetV1/Conv2d_3_depthwise/BatchNorm/AssignMovingAvg_1AssignSubVariableOpetrain_mobilenetv1_mobilenetv1_conv2d_3_depthwise_batchnorm_fusedbatchnormv3_readvariableop_1_resourceTtrain/MobilenetV1/MobilenetV1/Conv2d_3_depthwise/BatchNorm/AssignMovingAvg_1/mul:z:0*
_output_shapes
 *
dtype02N
Ltrain/MobilenetV1/MobilenetV1/Conv2d_3_depthwise/BatchNorm/AssignMovingAvg_1?
Ytrain/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/BatchNorm/AssignMovingAvg/ReadVariableOpReadVariableOpctrain_mobilenetv1_mobilenetv1_conv2d_3_pointwise_batchnorm_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:`*
dtype02[
Ytrain/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/BatchNorm/AssignMovingAvg/ReadVariableOp?
6train/MobilenetV1/MobilenetV1/Conv2d_3_depthwise/Relu6Relu6Otrain/MobilenetV1/MobilenetV1/Conv2d_3_depthwise/BatchNorm/FusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+???????????????????????????`28
6train/MobilenetV1/MobilenetV1/Conv2d_3_depthwise/Relu6?
Ftrain/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/Conv2D/ReadVariableOpReadVariableOpqtrain_mobilenetv1_mobilenetv1_conv2d_3_pointwise_kernel_regularizer_l2_regularizer_l2loss_readvariableop_resource*&
_output_shapes
:``*
dtype02H
Ftrain/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/Conv2D/ReadVariableOp?
7train/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/Conv2DConv2DDtrain/MobilenetV1/MobilenetV1/Conv2d_3_depthwise/Relu6:activations:0Ntrain/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????`*
paddingSAME*
strides
29
7train/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/Conv2D?
Itrain/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/BatchNorm/ReadVariableOpReadVariableOpRtrain_mobilenetv1_mobilenetv1_conv2d_3_pointwise_batchnorm_readvariableop_resource*
_output_shapes
:`*
dtype02K
Itrain/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/BatchNorm/ReadVariableOp?
Ktrain/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/BatchNorm/ReadVariableOp_1ReadVariableOpTtrain_mobilenetv1_mobilenetv1_conv2d_3_pointwise_batchnorm_readvariableop_1_resource*
_output_shapes
:`*
dtype02M
Ktrain/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/BatchNorm/ReadVariableOp_1?
Ztrain/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOpReadVariableOpctrain_mobilenetv1_mobilenetv1_conv2d_3_pointwise_batchnorm_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:`*
dtype02\
Ztrain/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp?
\train/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpetrain_mobilenetv1_mobilenetv1_conv2d_3_pointwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:`*
dtype02^
\train/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1?
Ktrain/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/BatchNorm/FusedBatchNormV3FusedBatchNormV3@train/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/Conv2D:output:0Qtrain/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/BatchNorm/ReadVariableOp:value:0Strain/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/BatchNorm/ReadVariableOp_1:value:0btrain/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp:value:0dtrain/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????`:`:`:`:`:*
epsilon%o?:2M
Ktrain/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/BatchNorm/FusedBatchNormV3?
Ptrain/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/BatchNorm/AssignMovingAvg/sub_1Subatrain/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/BatchNorm/AssignMovingAvg/ReadVariableOp:value:0Xtrain/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/BatchNorm/FusedBatchNormV3:batch_mean:0*
T0*
_output_shapes
:`2R
Ptrain/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/BatchNorm/AssignMovingAvg/sub_1?
Ptrain/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/BatchNorm/AssignMovingAvg/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2R
Ptrain/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/BatchNorm/AssignMovingAvg/sub/x?
Ntrain/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/BatchNorm/AssignMovingAvg/subSubYtrain/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/BatchNorm/AssignMovingAvg/sub/x:output:0placeholder_2*
T0*
_output_shapes
: 2P
Ntrain/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/BatchNorm/AssignMovingAvg/sub?
Ntrain/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/BatchNorm/AssignMovingAvg/mulMulTtrain/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/BatchNorm/AssignMovingAvg/sub_1:z:0Rtrain/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/BatchNorm/AssignMovingAvg/sub:z:0*
T0*
_output_shapes
:`2P
Ntrain/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/BatchNorm/AssignMovingAvg/mul?
Jtrain/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/BatchNorm/AssignMovingAvgAssignSubVariableOpctrain_mobilenetv1_mobilenetv1_conv2d_3_pointwise_batchnorm_fusedbatchnormv3_readvariableop_resourceRtrain/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/BatchNorm/AssignMovingAvg/mul:z:0*
_output_shapes
 *
dtype02L
Jtrain/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/BatchNorm/AssignMovingAvg?
[train/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/BatchNorm/AssignMovingAvg_1/ReadVariableOpReadVariableOpetrain_mobilenetv1_mobilenetv1_conv2d_3_pointwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:`*
dtype02]
[train/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/BatchNorm/AssignMovingAvg_1/ReadVariableOp?
Rtrain/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/BatchNorm/AssignMovingAvg_1/sub_1Subctrain/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/BatchNorm/AssignMovingAvg_1/ReadVariableOp:value:0\train/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/BatchNorm/FusedBatchNormV3:batch_variance:0*
T0*
_output_shapes
:`2T
Rtrain/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/BatchNorm/AssignMovingAvg_1/sub_1?
Rtrain/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/BatchNorm/AssignMovingAvg_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2T
Rtrain/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/BatchNorm/AssignMovingAvg_1/sub/x?
Ptrain/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/BatchNorm/AssignMovingAvg_1/subSub[train/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/BatchNorm/AssignMovingAvg_1/sub/x:output:0placeholder_2*
T0*
_output_shapes
: 2R
Ptrain/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/BatchNorm/AssignMovingAvg_1/sub?
Ptrain/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/BatchNorm/AssignMovingAvg_1/mulMulVtrain/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/BatchNorm/AssignMovingAvg_1/sub_1:z:0Ttrain/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/BatchNorm/AssignMovingAvg_1/sub:z:0*
T0*
_output_shapes
:`2R
Ptrain/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/BatchNorm/AssignMovingAvg_1/mul?
Ltrain/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/BatchNorm/AssignMovingAvg_1AssignSubVariableOpetrain_mobilenetv1_mobilenetv1_conv2d_3_pointwise_batchnorm_fusedbatchnormv3_readvariableop_1_resourceTtrain/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/BatchNorm/AssignMovingAvg_1/mul:z:0*
_output_shapes
 *
dtype02N
Ltrain/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/BatchNorm/AssignMovingAvg_1?
Ytrain/MobilenetV1/MobilenetV1/Conv2d_4_depthwise/BatchNorm/AssignMovingAvg/ReadVariableOpReadVariableOpctrain_mobilenetv1_mobilenetv1_conv2d_4_depthwise_batchnorm_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:`*
dtype02[
Ytrain/MobilenetV1/MobilenetV1/Conv2d_4_depthwise/BatchNorm/AssignMovingAvg/ReadVariableOp?
6train/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/Relu6Relu6Otrain/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/BatchNorm/FusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+???????????????????????????`28
6train/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/Relu6?
Itrain/MobilenetV1/MobilenetV1/Conv2d_4_depthwise/depthwise/ReadVariableOpReadVariableOpRtrain_mobilenetv1_mobilenetv1_conv2d_4_depthwise_depthwise_readvariableop_resource*&
_output_shapes
:`*
dtype02K
Itrain/MobilenetV1/MobilenetV1/Conv2d_4_depthwise/depthwise/ReadVariableOp?
:train/MobilenetV1/MobilenetV1/Conv2d_4_depthwise/depthwiseDepthwiseConv2dNativeDtrain/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/Relu6:activations:0Qtrain/MobilenetV1/MobilenetV1/Conv2d_4_depthwise/depthwise/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????`*
paddingSAME*
strides
2<
:train/MobilenetV1/MobilenetV1/Conv2d_4_depthwise/depthwise?
Itrain/MobilenetV1/MobilenetV1/Conv2d_4_depthwise/BatchNorm/ReadVariableOpReadVariableOpRtrain_mobilenetv1_mobilenetv1_conv2d_4_depthwise_batchnorm_readvariableop_resource*
_output_shapes
:`*
dtype02K
Itrain/MobilenetV1/MobilenetV1/Conv2d_4_depthwise/BatchNorm/ReadVariableOp?
Ktrain/MobilenetV1/MobilenetV1/Conv2d_4_depthwise/BatchNorm/ReadVariableOp_1ReadVariableOpTtrain_mobilenetv1_mobilenetv1_conv2d_4_depthwise_batchnorm_readvariableop_1_resource*
_output_shapes
:`*
dtype02M
Ktrain/MobilenetV1/MobilenetV1/Conv2d_4_depthwise/BatchNorm/ReadVariableOp_1?
Ztrain/MobilenetV1/MobilenetV1/Conv2d_4_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOpReadVariableOpctrain_mobilenetv1_mobilenetv1_conv2d_4_depthwise_batchnorm_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:`*
dtype02\
Ztrain/MobilenetV1/MobilenetV1/Conv2d_4_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp?
\train/MobilenetV1/MobilenetV1/Conv2d_4_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpetrain_mobilenetv1_mobilenetv1_conv2d_4_depthwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:`*
dtype02^
\train/MobilenetV1/MobilenetV1/Conv2d_4_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1?
Ktrain/MobilenetV1/MobilenetV1/Conv2d_4_depthwise/BatchNorm/FusedBatchNormV3FusedBatchNormV3Ctrain/MobilenetV1/MobilenetV1/Conv2d_4_depthwise/depthwise:output:0Qtrain/MobilenetV1/MobilenetV1/Conv2d_4_depthwise/BatchNorm/ReadVariableOp:value:0Strain/MobilenetV1/MobilenetV1/Conv2d_4_depthwise/BatchNorm/ReadVariableOp_1:value:0btrain/MobilenetV1/MobilenetV1/Conv2d_4_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp:value:0dtrain/MobilenetV1/MobilenetV1/Conv2d_4_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????`:`:`:`:`:*
epsilon%o?:2M
Ktrain/MobilenetV1/MobilenetV1/Conv2d_4_depthwise/BatchNorm/FusedBatchNormV3?
Ptrain/MobilenetV1/MobilenetV1/Conv2d_4_depthwise/BatchNorm/AssignMovingAvg/sub_1Subatrain/MobilenetV1/MobilenetV1/Conv2d_4_depthwise/BatchNorm/AssignMovingAvg/ReadVariableOp:value:0Xtrain/MobilenetV1/MobilenetV1/Conv2d_4_depthwise/BatchNorm/FusedBatchNormV3:batch_mean:0*
T0*
_output_shapes
:`2R
Ptrain/MobilenetV1/MobilenetV1/Conv2d_4_depthwise/BatchNorm/AssignMovingAvg/sub_1?
Ptrain/MobilenetV1/MobilenetV1/Conv2d_4_depthwise/BatchNorm/AssignMovingAvg/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2R
Ptrain/MobilenetV1/MobilenetV1/Conv2d_4_depthwise/BatchNorm/AssignMovingAvg/sub/x?
Ntrain/MobilenetV1/MobilenetV1/Conv2d_4_depthwise/BatchNorm/AssignMovingAvg/subSubYtrain/MobilenetV1/MobilenetV1/Conv2d_4_depthwise/BatchNorm/AssignMovingAvg/sub/x:output:0placeholder_2*
T0*
_output_shapes
: 2P
Ntrain/MobilenetV1/MobilenetV1/Conv2d_4_depthwise/BatchNorm/AssignMovingAvg/sub?
Ntrain/MobilenetV1/MobilenetV1/Conv2d_4_depthwise/BatchNorm/AssignMovingAvg/mulMulTtrain/MobilenetV1/MobilenetV1/Conv2d_4_depthwise/BatchNorm/AssignMovingAvg/sub_1:z:0Rtrain/MobilenetV1/MobilenetV1/Conv2d_4_depthwise/BatchNorm/AssignMovingAvg/sub:z:0*
T0*
_output_shapes
:`2P
Ntrain/MobilenetV1/MobilenetV1/Conv2d_4_depthwise/BatchNorm/AssignMovingAvg/mul?
Jtrain/MobilenetV1/MobilenetV1/Conv2d_4_depthwise/BatchNorm/AssignMovingAvgAssignSubVariableOpctrain_mobilenetv1_mobilenetv1_conv2d_4_depthwise_batchnorm_fusedbatchnormv3_readvariableop_resourceRtrain/MobilenetV1/MobilenetV1/Conv2d_4_depthwise/BatchNorm/AssignMovingAvg/mul:z:0*
_output_shapes
 *
dtype02L
Jtrain/MobilenetV1/MobilenetV1/Conv2d_4_depthwise/BatchNorm/AssignMovingAvg?
[train/MobilenetV1/MobilenetV1/Conv2d_4_depthwise/BatchNorm/AssignMovingAvg_1/ReadVariableOpReadVariableOpetrain_mobilenetv1_mobilenetv1_conv2d_4_depthwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:`*
dtype02]
[train/MobilenetV1/MobilenetV1/Conv2d_4_depthwise/BatchNorm/AssignMovingAvg_1/ReadVariableOp?
Rtrain/MobilenetV1/MobilenetV1/Conv2d_4_depthwise/BatchNorm/AssignMovingAvg_1/sub_1Subctrain/MobilenetV1/MobilenetV1/Conv2d_4_depthwise/BatchNorm/AssignMovingAvg_1/ReadVariableOp:value:0\train/MobilenetV1/MobilenetV1/Conv2d_4_depthwise/BatchNorm/FusedBatchNormV3:batch_variance:0*
T0*
_output_shapes
:`2T
Rtrain/MobilenetV1/MobilenetV1/Conv2d_4_depthwise/BatchNorm/AssignMovingAvg_1/sub_1?
Rtrain/MobilenetV1/MobilenetV1/Conv2d_4_depthwise/BatchNorm/AssignMovingAvg_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2T
Rtrain/MobilenetV1/MobilenetV1/Conv2d_4_depthwise/BatchNorm/AssignMovingAvg_1/sub/x?
Ptrain/MobilenetV1/MobilenetV1/Conv2d_4_depthwise/BatchNorm/AssignMovingAvg_1/subSub[train/MobilenetV1/MobilenetV1/Conv2d_4_depthwise/BatchNorm/AssignMovingAvg_1/sub/x:output:0placeholder_2*
T0*
_output_shapes
: 2R
Ptrain/MobilenetV1/MobilenetV1/Conv2d_4_depthwise/BatchNorm/AssignMovingAvg_1/sub?
Ptrain/MobilenetV1/MobilenetV1/Conv2d_4_depthwise/BatchNorm/AssignMovingAvg_1/mulMulVtrain/MobilenetV1/MobilenetV1/Conv2d_4_depthwise/BatchNorm/AssignMovingAvg_1/sub_1:z:0Ttrain/MobilenetV1/MobilenetV1/Conv2d_4_depthwise/BatchNorm/AssignMovingAvg_1/sub:z:0*
T0*
_output_shapes
:`2R
Ptrain/MobilenetV1/MobilenetV1/Conv2d_4_depthwise/BatchNorm/AssignMovingAvg_1/mul?
Ltrain/MobilenetV1/MobilenetV1/Conv2d_4_depthwise/BatchNorm/AssignMovingAvg_1AssignSubVariableOpetrain_mobilenetv1_mobilenetv1_conv2d_4_depthwise_batchnorm_fusedbatchnormv3_readvariableop_1_resourceTtrain/MobilenetV1/MobilenetV1/Conv2d_4_depthwise/BatchNorm/AssignMovingAvg_1/mul:z:0*
_output_shapes
 *
dtype02N
Ltrain/MobilenetV1/MobilenetV1/Conv2d_4_depthwise/BatchNorm/AssignMovingAvg_1?
Ytrain/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/BatchNorm/AssignMovingAvg/ReadVariableOpReadVariableOpctrain_mobilenetv1_mobilenetv1_conv2d_4_pointwise_batchnorm_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02[
Ytrain/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/BatchNorm/AssignMovingAvg/ReadVariableOp?
6train/MobilenetV1/MobilenetV1/Conv2d_4_depthwise/Relu6Relu6Otrain/MobilenetV1/MobilenetV1/Conv2d_4_depthwise/BatchNorm/FusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+???????????????????????????`28
6train/MobilenetV1/MobilenetV1/Conv2d_4_depthwise/Relu6?
Ftrain/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/Conv2D/ReadVariableOpReadVariableOpqtrain_mobilenetv1_mobilenetv1_conv2d_4_pointwise_kernel_regularizer_l2_regularizer_l2loss_readvariableop_resource*'
_output_shapes
:`?*
dtype02H
Ftrain/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/Conv2D/ReadVariableOp?
7train/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/Conv2DConv2DDtrain/MobilenetV1/MobilenetV1/Conv2d_4_depthwise/Relu6:activations:0Ntrain/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
29
7train/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/Conv2D?
Itrain/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/BatchNorm/ReadVariableOpReadVariableOpRtrain_mobilenetv1_mobilenetv1_conv2d_4_pointwise_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02K
Itrain/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/BatchNorm/ReadVariableOp?
Ktrain/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/BatchNorm/ReadVariableOp_1ReadVariableOpTtrain_mobilenetv1_mobilenetv1_conv2d_4_pointwise_batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype02M
Ktrain/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/BatchNorm/ReadVariableOp_1?
Ztrain/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOpReadVariableOpctrain_mobilenetv1_mobilenetv1_conv2d_4_pointwise_batchnorm_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02\
Ztrain/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp?
\train/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpetrain_mobilenetv1_mobilenetv1_conv2d_4_pointwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02^
\train/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1?
Ktrain/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/BatchNorm/FusedBatchNormV3FusedBatchNormV3@train/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/Conv2D:output:0Qtrain/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/BatchNorm/ReadVariableOp:value:0Strain/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/BatchNorm/ReadVariableOp_1:value:0btrain/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp:value:0dtrain/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:2M
Ktrain/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/BatchNorm/FusedBatchNormV3?
Ptrain/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/BatchNorm/AssignMovingAvg/sub_1Subatrain/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/BatchNorm/AssignMovingAvg/ReadVariableOp:value:0Xtrain/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/BatchNorm/FusedBatchNormV3:batch_mean:0*
T0*
_output_shapes	
:?2R
Ptrain/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/BatchNorm/AssignMovingAvg/sub_1?
Ptrain/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/BatchNorm/AssignMovingAvg/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2R
Ptrain/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/BatchNorm/AssignMovingAvg/sub/x?
Ntrain/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/BatchNorm/AssignMovingAvg/subSubYtrain/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/BatchNorm/AssignMovingAvg/sub/x:output:0placeholder_2*
T0*
_output_shapes
: 2P
Ntrain/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/BatchNorm/AssignMovingAvg/sub?
Ntrain/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/BatchNorm/AssignMovingAvg/mulMulTtrain/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/BatchNorm/AssignMovingAvg/sub_1:z:0Rtrain/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/BatchNorm/AssignMovingAvg/sub:z:0*
T0*
_output_shapes	
:?2P
Ntrain/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/BatchNorm/AssignMovingAvg/mul?
Jtrain/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/BatchNorm/AssignMovingAvgAssignSubVariableOpctrain_mobilenetv1_mobilenetv1_conv2d_4_pointwise_batchnorm_fusedbatchnormv3_readvariableop_resourceRtrain/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/BatchNorm/AssignMovingAvg/mul:z:0*
_output_shapes
 *
dtype02L
Jtrain/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/BatchNorm/AssignMovingAvg?
[train/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/BatchNorm/AssignMovingAvg_1/ReadVariableOpReadVariableOpetrain_mobilenetv1_mobilenetv1_conv2d_4_pointwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02]
[train/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/BatchNorm/AssignMovingAvg_1/ReadVariableOp?
Rtrain/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/BatchNorm/AssignMovingAvg_1/sub_1Subctrain/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/BatchNorm/AssignMovingAvg_1/ReadVariableOp:value:0\train/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/BatchNorm/FusedBatchNormV3:batch_variance:0*
T0*
_output_shapes	
:?2T
Rtrain/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/BatchNorm/AssignMovingAvg_1/sub_1?
Rtrain/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/BatchNorm/AssignMovingAvg_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2T
Rtrain/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/BatchNorm/AssignMovingAvg_1/sub/x?
Ptrain/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/BatchNorm/AssignMovingAvg_1/subSub[train/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/BatchNorm/AssignMovingAvg_1/sub/x:output:0placeholder_2*
T0*
_output_shapes
: 2R
Ptrain/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/BatchNorm/AssignMovingAvg_1/sub?
Ptrain/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/BatchNorm/AssignMovingAvg_1/mulMulVtrain/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/BatchNorm/AssignMovingAvg_1/sub_1:z:0Ttrain/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/BatchNorm/AssignMovingAvg_1/sub:z:0*
T0*
_output_shapes	
:?2R
Ptrain/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/BatchNorm/AssignMovingAvg_1/mul?
Ltrain/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/BatchNorm/AssignMovingAvg_1AssignSubVariableOpetrain_mobilenetv1_mobilenetv1_conv2d_4_pointwise_batchnorm_fusedbatchnormv3_readvariableop_1_resourceTtrain/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/BatchNorm/AssignMovingAvg_1/mul:z:0*
_output_shapes
 *
dtype02N
Ltrain/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/BatchNorm/AssignMovingAvg_1?
Ytrain/MobilenetV1/MobilenetV1/Conv2d_5_depthwise/BatchNorm/AssignMovingAvg/ReadVariableOpReadVariableOpctrain_mobilenetv1_mobilenetv1_conv2d_5_depthwise_batchnorm_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02[
Ytrain/MobilenetV1/MobilenetV1/Conv2d_5_depthwise/BatchNorm/AssignMovingAvg/ReadVariableOp?
6train/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/Relu6Relu6Otrain/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/BatchNorm/FusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,????????????????????????????28
6train/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/Relu6?
Itrain/MobilenetV1/MobilenetV1/Conv2d_5_depthwise/depthwise/ReadVariableOpReadVariableOpRtrain_mobilenetv1_mobilenetv1_conv2d_5_depthwise_depthwise_readvariableop_resource*'
_output_shapes
:?*
dtype02K
Itrain/MobilenetV1/MobilenetV1/Conv2d_5_depthwise/depthwise/ReadVariableOp?
:train/MobilenetV1/MobilenetV1/Conv2d_5_depthwise/depthwiseDepthwiseConv2dNativeDtrain/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/Relu6:activations:0Qtrain/MobilenetV1/MobilenetV1/Conv2d_5_depthwise/depthwise/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
2<
:train/MobilenetV1/MobilenetV1/Conv2d_5_depthwise/depthwise?
Itrain/MobilenetV1/MobilenetV1/Conv2d_5_depthwise/BatchNorm/ReadVariableOpReadVariableOpRtrain_mobilenetv1_mobilenetv1_conv2d_5_depthwise_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02K
Itrain/MobilenetV1/MobilenetV1/Conv2d_5_depthwise/BatchNorm/ReadVariableOp?
Ktrain/MobilenetV1/MobilenetV1/Conv2d_5_depthwise/BatchNorm/ReadVariableOp_1ReadVariableOpTtrain_mobilenetv1_mobilenetv1_conv2d_5_depthwise_batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype02M
Ktrain/MobilenetV1/MobilenetV1/Conv2d_5_depthwise/BatchNorm/ReadVariableOp_1?
Ztrain/MobilenetV1/MobilenetV1/Conv2d_5_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOpReadVariableOpctrain_mobilenetv1_mobilenetv1_conv2d_5_depthwise_batchnorm_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02\
Ztrain/MobilenetV1/MobilenetV1/Conv2d_5_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp?
\train/MobilenetV1/MobilenetV1/Conv2d_5_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpetrain_mobilenetv1_mobilenetv1_conv2d_5_depthwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02^
\train/MobilenetV1/MobilenetV1/Conv2d_5_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1?
Ktrain/MobilenetV1/MobilenetV1/Conv2d_5_depthwise/BatchNorm/FusedBatchNormV3FusedBatchNormV3Ctrain/MobilenetV1/MobilenetV1/Conv2d_5_depthwise/depthwise:output:0Qtrain/MobilenetV1/MobilenetV1/Conv2d_5_depthwise/BatchNorm/ReadVariableOp:value:0Strain/MobilenetV1/MobilenetV1/Conv2d_5_depthwise/BatchNorm/ReadVariableOp_1:value:0btrain/MobilenetV1/MobilenetV1/Conv2d_5_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp:value:0dtrain/MobilenetV1/MobilenetV1/Conv2d_5_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:2M
Ktrain/MobilenetV1/MobilenetV1/Conv2d_5_depthwise/BatchNorm/FusedBatchNormV3?
Ptrain/MobilenetV1/MobilenetV1/Conv2d_5_depthwise/BatchNorm/AssignMovingAvg/sub_1Subatrain/MobilenetV1/MobilenetV1/Conv2d_5_depthwise/BatchNorm/AssignMovingAvg/ReadVariableOp:value:0Xtrain/MobilenetV1/MobilenetV1/Conv2d_5_depthwise/BatchNorm/FusedBatchNormV3:batch_mean:0*
T0*
_output_shapes	
:?2R
Ptrain/MobilenetV1/MobilenetV1/Conv2d_5_depthwise/BatchNorm/AssignMovingAvg/sub_1?
Ptrain/MobilenetV1/MobilenetV1/Conv2d_5_depthwise/BatchNorm/AssignMovingAvg/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2R
Ptrain/MobilenetV1/MobilenetV1/Conv2d_5_depthwise/BatchNorm/AssignMovingAvg/sub/x?
Ntrain/MobilenetV1/MobilenetV1/Conv2d_5_depthwise/BatchNorm/AssignMovingAvg/subSubYtrain/MobilenetV1/MobilenetV1/Conv2d_5_depthwise/BatchNorm/AssignMovingAvg/sub/x:output:0placeholder_2*
T0*
_output_shapes
: 2P
Ntrain/MobilenetV1/MobilenetV1/Conv2d_5_depthwise/BatchNorm/AssignMovingAvg/sub?
Ntrain/MobilenetV1/MobilenetV1/Conv2d_5_depthwise/BatchNorm/AssignMovingAvg/mulMulTtrain/MobilenetV1/MobilenetV1/Conv2d_5_depthwise/BatchNorm/AssignMovingAvg/sub_1:z:0Rtrain/MobilenetV1/MobilenetV1/Conv2d_5_depthwise/BatchNorm/AssignMovingAvg/sub:z:0*
T0*
_output_shapes	
:?2P
Ntrain/MobilenetV1/MobilenetV1/Conv2d_5_depthwise/BatchNorm/AssignMovingAvg/mul?
Jtrain/MobilenetV1/MobilenetV1/Conv2d_5_depthwise/BatchNorm/AssignMovingAvgAssignSubVariableOpctrain_mobilenetv1_mobilenetv1_conv2d_5_depthwise_batchnorm_fusedbatchnormv3_readvariableop_resourceRtrain/MobilenetV1/MobilenetV1/Conv2d_5_depthwise/BatchNorm/AssignMovingAvg/mul:z:0*
_output_shapes
 *
dtype02L
Jtrain/MobilenetV1/MobilenetV1/Conv2d_5_depthwise/BatchNorm/AssignMovingAvg?
[train/MobilenetV1/MobilenetV1/Conv2d_5_depthwise/BatchNorm/AssignMovingAvg_1/ReadVariableOpReadVariableOpetrain_mobilenetv1_mobilenetv1_conv2d_5_depthwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02]
[train/MobilenetV1/MobilenetV1/Conv2d_5_depthwise/BatchNorm/AssignMovingAvg_1/ReadVariableOp?
Rtrain/MobilenetV1/MobilenetV1/Conv2d_5_depthwise/BatchNorm/AssignMovingAvg_1/sub_1Subctrain/MobilenetV1/MobilenetV1/Conv2d_5_depthwise/BatchNorm/AssignMovingAvg_1/ReadVariableOp:value:0\train/MobilenetV1/MobilenetV1/Conv2d_5_depthwise/BatchNorm/FusedBatchNormV3:batch_variance:0*
T0*
_output_shapes	
:?2T
Rtrain/MobilenetV1/MobilenetV1/Conv2d_5_depthwise/BatchNorm/AssignMovingAvg_1/sub_1?
Rtrain/MobilenetV1/MobilenetV1/Conv2d_5_depthwise/BatchNorm/AssignMovingAvg_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2T
Rtrain/MobilenetV1/MobilenetV1/Conv2d_5_depthwise/BatchNorm/AssignMovingAvg_1/sub/x?
Ptrain/MobilenetV1/MobilenetV1/Conv2d_5_depthwise/BatchNorm/AssignMovingAvg_1/subSub[train/MobilenetV1/MobilenetV1/Conv2d_5_depthwise/BatchNorm/AssignMovingAvg_1/sub/x:output:0placeholder_2*
T0*
_output_shapes
: 2R
Ptrain/MobilenetV1/MobilenetV1/Conv2d_5_depthwise/BatchNorm/AssignMovingAvg_1/sub?
Ptrain/MobilenetV1/MobilenetV1/Conv2d_5_depthwise/BatchNorm/AssignMovingAvg_1/mulMulVtrain/MobilenetV1/MobilenetV1/Conv2d_5_depthwise/BatchNorm/AssignMovingAvg_1/sub_1:z:0Ttrain/MobilenetV1/MobilenetV1/Conv2d_5_depthwise/BatchNorm/AssignMovingAvg_1/sub:z:0*
T0*
_output_shapes	
:?2R
Ptrain/MobilenetV1/MobilenetV1/Conv2d_5_depthwise/BatchNorm/AssignMovingAvg_1/mul?
Ltrain/MobilenetV1/MobilenetV1/Conv2d_5_depthwise/BatchNorm/AssignMovingAvg_1AssignSubVariableOpetrain_mobilenetv1_mobilenetv1_conv2d_5_depthwise_batchnorm_fusedbatchnormv3_readvariableop_1_resourceTtrain/MobilenetV1/MobilenetV1/Conv2d_5_depthwise/BatchNorm/AssignMovingAvg_1/mul:z:0*
_output_shapes
 *
dtype02N
Ltrain/MobilenetV1/MobilenetV1/Conv2d_5_depthwise/BatchNorm/AssignMovingAvg_1?
Ytrain/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/BatchNorm/AssignMovingAvg/ReadVariableOpReadVariableOpctrain_mobilenetv1_mobilenetv1_conv2d_5_pointwise_batchnorm_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02[
Ytrain/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/BatchNorm/AssignMovingAvg/ReadVariableOp?
6train/MobilenetV1/MobilenetV1/Conv2d_5_depthwise/Relu6Relu6Otrain/MobilenetV1/MobilenetV1/Conv2d_5_depthwise/BatchNorm/FusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,????????????????????????????28
6train/MobilenetV1/MobilenetV1/Conv2d_5_depthwise/Relu6?
Ftrain/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/Conv2D/ReadVariableOpReadVariableOpqtrain_mobilenetv1_mobilenetv1_conv2d_5_pointwise_kernel_regularizer_l2_regularizer_l2loss_readvariableop_resource*(
_output_shapes
:??*
dtype02H
Ftrain/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/Conv2D/ReadVariableOp?
7train/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/Conv2DConv2DDtrain/MobilenetV1/MobilenetV1/Conv2d_5_depthwise/Relu6:activations:0Ntrain/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
29
7train/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/Conv2D?
Itrain/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/BatchNorm/ReadVariableOpReadVariableOpRtrain_mobilenetv1_mobilenetv1_conv2d_5_pointwise_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02K
Itrain/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/BatchNorm/ReadVariableOp?
Ktrain/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/BatchNorm/ReadVariableOp_1ReadVariableOpTtrain_mobilenetv1_mobilenetv1_conv2d_5_pointwise_batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype02M
Ktrain/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/BatchNorm/ReadVariableOp_1?
Ztrain/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOpReadVariableOpctrain_mobilenetv1_mobilenetv1_conv2d_5_pointwise_batchnorm_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02\
Ztrain/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp?
\train/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpetrain_mobilenetv1_mobilenetv1_conv2d_5_pointwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02^
\train/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1?
Ktrain/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/BatchNorm/FusedBatchNormV3FusedBatchNormV3@train/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/Conv2D:output:0Qtrain/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/BatchNorm/ReadVariableOp:value:0Strain/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/BatchNorm/ReadVariableOp_1:value:0btrain/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp:value:0dtrain/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:2M
Ktrain/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/BatchNorm/FusedBatchNormV3?
Ptrain/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/BatchNorm/AssignMovingAvg/sub_1Subatrain/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/BatchNorm/AssignMovingAvg/ReadVariableOp:value:0Xtrain/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/BatchNorm/FusedBatchNormV3:batch_mean:0*
T0*
_output_shapes	
:?2R
Ptrain/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/BatchNorm/AssignMovingAvg/sub_1?
Ptrain/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/BatchNorm/AssignMovingAvg/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2R
Ptrain/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/BatchNorm/AssignMovingAvg/sub/x?
Ntrain/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/BatchNorm/AssignMovingAvg/subSubYtrain/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/BatchNorm/AssignMovingAvg/sub/x:output:0placeholder_2*
T0*
_output_shapes
: 2P
Ntrain/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/BatchNorm/AssignMovingAvg/sub?
Ntrain/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/BatchNorm/AssignMovingAvg/mulMulTtrain/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/BatchNorm/AssignMovingAvg/sub_1:z:0Rtrain/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/BatchNorm/AssignMovingAvg/sub:z:0*
T0*
_output_shapes	
:?2P
Ntrain/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/BatchNorm/AssignMovingAvg/mul?
Jtrain/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/BatchNorm/AssignMovingAvgAssignSubVariableOpctrain_mobilenetv1_mobilenetv1_conv2d_5_pointwise_batchnorm_fusedbatchnormv3_readvariableop_resourceRtrain/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/BatchNorm/AssignMovingAvg/mul:z:0*
_output_shapes
 *
dtype02L
Jtrain/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/BatchNorm/AssignMovingAvg?
[train/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/BatchNorm/AssignMovingAvg_1/ReadVariableOpReadVariableOpetrain_mobilenetv1_mobilenetv1_conv2d_5_pointwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02]
[train/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/BatchNorm/AssignMovingAvg_1/ReadVariableOp?
Rtrain/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/BatchNorm/AssignMovingAvg_1/sub_1Subctrain/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/BatchNorm/AssignMovingAvg_1/ReadVariableOp:value:0\train/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/BatchNorm/FusedBatchNormV3:batch_variance:0*
T0*
_output_shapes	
:?2T
Rtrain/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/BatchNorm/AssignMovingAvg_1/sub_1?
Rtrain/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/BatchNorm/AssignMovingAvg_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2T
Rtrain/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/BatchNorm/AssignMovingAvg_1/sub/x?
Ptrain/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/BatchNorm/AssignMovingAvg_1/subSub[train/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/BatchNorm/AssignMovingAvg_1/sub/x:output:0placeholder_2*
T0*
_output_shapes
: 2R
Ptrain/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/BatchNorm/AssignMovingAvg_1/sub?
Ptrain/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/BatchNorm/AssignMovingAvg_1/mulMulVtrain/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/BatchNorm/AssignMovingAvg_1/sub_1:z:0Ttrain/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/BatchNorm/AssignMovingAvg_1/sub:z:0*
T0*
_output_shapes	
:?2R
Ptrain/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/BatchNorm/AssignMovingAvg_1/mul?
Ltrain/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/BatchNorm/AssignMovingAvg_1AssignSubVariableOpetrain_mobilenetv1_mobilenetv1_conv2d_5_pointwise_batchnorm_fusedbatchnormv3_readvariableop_1_resourceTtrain/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/BatchNorm/AssignMovingAvg_1/mul:z:0*
_output_shapes
 *
dtype02N
Ltrain/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/BatchNorm/AssignMovingAvg_1?
Ytrain/MobilenetV1/MobilenetV1/Conv2d_6_depthwise/BatchNorm/AssignMovingAvg/ReadVariableOpReadVariableOpctrain_mobilenetv1_mobilenetv1_conv2d_6_depthwise_batchnorm_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02[
Ytrain/MobilenetV1/MobilenetV1/Conv2d_6_depthwise/BatchNorm/AssignMovingAvg/ReadVariableOp?
6train/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/Relu6Relu6Otrain/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/BatchNorm/FusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,????????????????????????????28
6train/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/Relu6?
Itrain/MobilenetV1/MobilenetV1/Conv2d_6_depthwise/depthwise/ReadVariableOpReadVariableOpRtrain_mobilenetv1_mobilenetv1_conv2d_6_depthwise_depthwise_readvariableop_resource*'
_output_shapes
:?*
dtype02K
Itrain/MobilenetV1/MobilenetV1/Conv2d_6_depthwise/depthwise/ReadVariableOp?
:train/MobilenetV1/MobilenetV1/Conv2d_6_depthwise/depthwiseDepthwiseConv2dNativeDtrain/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/Relu6:activations:0Qtrain/MobilenetV1/MobilenetV1/Conv2d_6_depthwise/depthwise/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
2<
:train/MobilenetV1/MobilenetV1/Conv2d_6_depthwise/depthwise?
Itrain/MobilenetV1/MobilenetV1/Conv2d_6_depthwise/BatchNorm/ReadVariableOpReadVariableOpRtrain_mobilenetv1_mobilenetv1_conv2d_6_depthwise_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02K
Itrain/MobilenetV1/MobilenetV1/Conv2d_6_depthwise/BatchNorm/ReadVariableOp?
Ktrain/MobilenetV1/MobilenetV1/Conv2d_6_depthwise/BatchNorm/ReadVariableOp_1ReadVariableOpTtrain_mobilenetv1_mobilenetv1_conv2d_6_depthwise_batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype02M
Ktrain/MobilenetV1/MobilenetV1/Conv2d_6_depthwise/BatchNorm/ReadVariableOp_1?
Ztrain/MobilenetV1/MobilenetV1/Conv2d_6_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOpReadVariableOpctrain_mobilenetv1_mobilenetv1_conv2d_6_depthwise_batchnorm_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02\
Ztrain/MobilenetV1/MobilenetV1/Conv2d_6_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp?
\train/MobilenetV1/MobilenetV1/Conv2d_6_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpetrain_mobilenetv1_mobilenetv1_conv2d_6_depthwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02^
\train/MobilenetV1/MobilenetV1/Conv2d_6_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1?
Ktrain/MobilenetV1/MobilenetV1/Conv2d_6_depthwise/BatchNorm/FusedBatchNormV3FusedBatchNormV3Ctrain/MobilenetV1/MobilenetV1/Conv2d_6_depthwise/depthwise:output:0Qtrain/MobilenetV1/MobilenetV1/Conv2d_6_depthwise/BatchNorm/ReadVariableOp:value:0Strain/MobilenetV1/MobilenetV1/Conv2d_6_depthwise/BatchNorm/ReadVariableOp_1:value:0btrain/MobilenetV1/MobilenetV1/Conv2d_6_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp:value:0dtrain/MobilenetV1/MobilenetV1/Conv2d_6_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:2M
Ktrain/MobilenetV1/MobilenetV1/Conv2d_6_depthwise/BatchNorm/FusedBatchNormV3?
Ptrain/MobilenetV1/MobilenetV1/Conv2d_6_depthwise/BatchNorm/AssignMovingAvg/sub_1Subatrain/MobilenetV1/MobilenetV1/Conv2d_6_depthwise/BatchNorm/AssignMovingAvg/ReadVariableOp:value:0Xtrain/MobilenetV1/MobilenetV1/Conv2d_6_depthwise/BatchNorm/FusedBatchNormV3:batch_mean:0*
T0*
_output_shapes	
:?2R
Ptrain/MobilenetV1/MobilenetV1/Conv2d_6_depthwise/BatchNorm/AssignMovingAvg/sub_1?
Ptrain/MobilenetV1/MobilenetV1/Conv2d_6_depthwise/BatchNorm/AssignMovingAvg/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2R
Ptrain/MobilenetV1/MobilenetV1/Conv2d_6_depthwise/BatchNorm/AssignMovingAvg/sub/x?
Ntrain/MobilenetV1/MobilenetV1/Conv2d_6_depthwise/BatchNorm/AssignMovingAvg/subSubYtrain/MobilenetV1/MobilenetV1/Conv2d_6_depthwise/BatchNorm/AssignMovingAvg/sub/x:output:0placeholder_2*
T0*
_output_shapes
: 2P
Ntrain/MobilenetV1/MobilenetV1/Conv2d_6_depthwise/BatchNorm/AssignMovingAvg/sub?
Ntrain/MobilenetV1/MobilenetV1/Conv2d_6_depthwise/BatchNorm/AssignMovingAvg/mulMulTtrain/MobilenetV1/MobilenetV1/Conv2d_6_depthwise/BatchNorm/AssignMovingAvg/sub_1:z:0Rtrain/MobilenetV1/MobilenetV1/Conv2d_6_depthwise/BatchNorm/AssignMovingAvg/sub:z:0*
T0*
_output_shapes	
:?2P
Ntrain/MobilenetV1/MobilenetV1/Conv2d_6_depthwise/BatchNorm/AssignMovingAvg/mul?
Jtrain/MobilenetV1/MobilenetV1/Conv2d_6_depthwise/BatchNorm/AssignMovingAvgAssignSubVariableOpctrain_mobilenetv1_mobilenetv1_conv2d_6_depthwise_batchnorm_fusedbatchnormv3_readvariableop_resourceRtrain/MobilenetV1/MobilenetV1/Conv2d_6_depthwise/BatchNorm/AssignMovingAvg/mul:z:0*
_output_shapes
 *
dtype02L
Jtrain/MobilenetV1/MobilenetV1/Conv2d_6_depthwise/BatchNorm/AssignMovingAvg?
[train/MobilenetV1/MobilenetV1/Conv2d_6_depthwise/BatchNorm/AssignMovingAvg_1/ReadVariableOpReadVariableOpetrain_mobilenetv1_mobilenetv1_conv2d_6_depthwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02]
[train/MobilenetV1/MobilenetV1/Conv2d_6_depthwise/BatchNorm/AssignMovingAvg_1/ReadVariableOp?
Rtrain/MobilenetV1/MobilenetV1/Conv2d_6_depthwise/BatchNorm/AssignMovingAvg_1/sub_1Subctrain/MobilenetV1/MobilenetV1/Conv2d_6_depthwise/BatchNorm/AssignMovingAvg_1/ReadVariableOp:value:0\train/MobilenetV1/MobilenetV1/Conv2d_6_depthwise/BatchNorm/FusedBatchNormV3:batch_variance:0*
T0*
_output_shapes	
:?2T
Rtrain/MobilenetV1/MobilenetV1/Conv2d_6_depthwise/BatchNorm/AssignMovingAvg_1/sub_1?
Rtrain/MobilenetV1/MobilenetV1/Conv2d_6_depthwise/BatchNorm/AssignMovingAvg_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2T
Rtrain/MobilenetV1/MobilenetV1/Conv2d_6_depthwise/BatchNorm/AssignMovingAvg_1/sub/x?
Ptrain/MobilenetV1/MobilenetV1/Conv2d_6_depthwise/BatchNorm/AssignMovingAvg_1/subSub[train/MobilenetV1/MobilenetV1/Conv2d_6_depthwise/BatchNorm/AssignMovingAvg_1/sub/x:output:0placeholder_2*
T0*
_output_shapes
: 2R
Ptrain/MobilenetV1/MobilenetV1/Conv2d_6_depthwise/BatchNorm/AssignMovingAvg_1/sub?
Ptrain/MobilenetV1/MobilenetV1/Conv2d_6_depthwise/BatchNorm/AssignMovingAvg_1/mulMulVtrain/MobilenetV1/MobilenetV1/Conv2d_6_depthwise/BatchNorm/AssignMovingAvg_1/sub_1:z:0Ttrain/MobilenetV1/MobilenetV1/Conv2d_6_depthwise/BatchNorm/AssignMovingAvg_1/sub:z:0*
T0*
_output_shapes	
:?2R
Ptrain/MobilenetV1/MobilenetV1/Conv2d_6_depthwise/BatchNorm/AssignMovingAvg_1/mul?
Ltrain/MobilenetV1/MobilenetV1/Conv2d_6_depthwise/BatchNorm/AssignMovingAvg_1AssignSubVariableOpetrain_mobilenetv1_mobilenetv1_conv2d_6_depthwise_batchnorm_fusedbatchnormv3_readvariableop_1_resourceTtrain/MobilenetV1/MobilenetV1/Conv2d_6_depthwise/BatchNorm/AssignMovingAvg_1/mul:z:0*
_output_shapes
 *
dtype02N
Ltrain/MobilenetV1/MobilenetV1/Conv2d_6_depthwise/BatchNorm/AssignMovingAvg_1?
Ytrain/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/BatchNorm/AssignMovingAvg/ReadVariableOpReadVariableOpctrain_mobilenetv1_mobilenetv1_conv2d_6_pointwise_batchnorm_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02[
Ytrain/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/BatchNorm/AssignMovingAvg/ReadVariableOp?
6train/MobilenetV1/MobilenetV1/Conv2d_6_depthwise/Relu6Relu6Otrain/MobilenetV1/MobilenetV1/Conv2d_6_depthwise/BatchNorm/FusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,????????????????????????????28
6train/MobilenetV1/MobilenetV1/Conv2d_6_depthwise/Relu6?
Ftrain/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/Conv2D/ReadVariableOpReadVariableOpqtrain_mobilenetv1_mobilenetv1_conv2d_6_pointwise_kernel_regularizer_l2_regularizer_l2loss_readvariableop_resource*(
_output_shapes
:??*
dtype02H
Ftrain/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/Conv2D/ReadVariableOp?
7train/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/Conv2DConv2DDtrain/MobilenetV1/MobilenetV1/Conv2d_6_depthwise/Relu6:activations:0Ntrain/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
29
7train/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/Conv2D?
Itrain/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/BatchNorm/ReadVariableOpReadVariableOpRtrain_mobilenetv1_mobilenetv1_conv2d_6_pointwise_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02K
Itrain/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/BatchNorm/ReadVariableOp?
Ktrain/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/BatchNorm/ReadVariableOp_1ReadVariableOpTtrain_mobilenetv1_mobilenetv1_conv2d_6_pointwise_batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype02M
Ktrain/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/BatchNorm/ReadVariableOp_1?
Ztrain/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOpReadVariableOpctrain_mobilenetv1_mobilenetv1_conv2d_6_pointwise_batchnorm_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02\
Ztrain/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp?
\train/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpetrain_mobilenetv1_mobilenetv1_conv2d_6_pointwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02^
\train/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1?
Ktrain/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/BatchNorm/FusedBatchNormV3FusedBatchNormV3@train/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/Conv2D:output:0Qtrain/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/BatchNorm/ReadVariableOp:value:0Strain/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/BatchNorm/ReadVariableOp_1:value:0btrain/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp:value:0dtrain/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:2M
Ktrain/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/BatchNorm/FusedBatchNormV3?
Ptrain/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/BatchNorm/AssignMovingAvg/sub_1Subatrain/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/BatchNorm/AssignMovingAvg/ReadVariableOp:value:0Xtrain/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/BatchNorm/FusedBatchNormV3:batch_mean:0*
T0*
_output_shapes	
:?2R
Ptrain/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/BatchNorm/AssignMovingAvg/sub_1?
Ptrain/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/BatchNorm/AssignMovingAvg/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2R
Ptrain/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/BatchNorm/AssignMovingAvg/sub/x?
Ntrain/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/BatchNorm/AssignMovingAvg/subSubYtrain/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/BatchNorm/AssignMovingAvg/sub/x:output:0placeholder_2*
T0*
_output_shapes
: 2P
Ntrain/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/BatchNorm/AssignMovingAvg/sub?
Ntrain/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/BatchNorm/AssignMovingAvg/mulMulTtrain/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/BatchNorm/AssignMovingAvg/sub_1:z:0Rtrain/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/BatchNorm/AssignMovingAvg/sub:z:0*
T0*
_output_shapes	
:?2P
Ntrain/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/BatchNorm/AssignMovingAvg/mul?
Jtrain/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/BatchNorm/AssignMovingAvgAssignSubVariableOpctrain_mobilenetv1_mobilenetv1_conv2d_6_pointwise_batchnorm_fusedbatchnormv3_readvariableop_resourceRtrain/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/BatchNorm/AssignMovingAvg/mul:z:0*
_output_shapes
 *
dtype02L
Jtrain/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/BatchNorm/AssignMovingAvg?
[train/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/BatchNorm/AssignMovingAvg_1/ReadVariableOpReadVariableOpetrain_mobilenetv1_mobilenetv1_conv2d_6_pointwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02]
[train/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/BatchNorm/AssignMovingAvg_1/ReadVariableOp?
Rtrain/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/BatchNorm/AssignMovingAvg_1/sub_1Subctrain/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/BatchNorm/AssignMovingAvg_1/ReadVariableOp:value:0\train/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/BatchNorm/FusedBatchNormV3:batch_variance:0*
T0*
_output_shapes	
:?2T
Rtrain/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/BatchNorm/AssignMovingAvg_1/sub_1?
Rtrain/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/BatchNorm/AssignMovingAvg_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2T
Rtrain/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/BatchNorm/AssignMovingAvg_1/sub/x?
Ptrain/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/BatchNorm/AssignMovingAvg_1/subSub[train/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/BatchNorm/AssignMovingAvg_1/sub/x:output:0placeholder_2*
T0*
_output_shapes
: 2R
Ptrain/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/BatchNorm/AssignMovingAvg_1/sub?
Ptrain/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/BatchNorm/AssignMovingAvg_1/mulMulVtrain/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/BatchNorm/AssignMovingAvg_1/sub_1:z:0Ttrain/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/BatchNorm/AssignMovingAvg_1/sub:z:0*
T0*
_output_shapes	
:?2R
Ptrain/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/BatchNorm/AssignMovingAvg_1/mul?
Ltrain/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/BatchNorm/AssignMovingAvg_1AssignSubVariableOpetrain_mobilenetv1_mobilenetv1_conv2d_6_pointwise_batchnorm_fusedbatchnormv3_readvariableop_1_resourceTtrain/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/BatchNorm/AssignMovingAvg_1/mul:z:0*
_output_shapes
 *
dtype02N
Ltrain/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/BatchNorm/AssignMovingAvg_1?
Ytrain/MobilenetV1/MobilenetV1/Conv2d_7_depthwise/BatchNorm/AssignMovingAvg/ReadVariableOpReadVariableOpctrain_mobilenetv1_mobilenetv1_conv2d_7_depthwise_batchnorm_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02[
Ytrain/MobilenetV1/MobilenetV1/Conv2d_7_depthwise/BatchNorm/AssignMovingAvg/ReadVariableOp?
6train/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/Relu6Relu6Otrain/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/BatchNorm/FusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,????????????????????????????28
6train/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/Relu6?
Itrain/MobilenetV1/MobilenetV1/Conv2d_7_depthwise/depthwise/ReadVariableOpReadVariableOpRtrain_mobilenetv1_mobilenetv1_conv2d_7_depthwise_depthwise_readvariableop_resource*'
_output_shapes
:?*
dtype02K
Itrain/MobilenetV1/MobilenetV1/Conv2d_7_depthwise/depthwise/ReadVariableOp?
:train/MobilenetV1/MobilenetV1/Conv2d_7_depthwise/depthwiseDepthwiseConv2dNativeDtrain/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/Relu6:activations:0Qtrain/MobilenetV1/MobilenetV1/Conv2d_7_depthwise/depthwise/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
2<
:train/MobilenetV1/MobilenetV1/Conv2d_7_depthwise/depthwise?
Itrain/MobilenetV1/MobilenetV1/Conv2d_7_depthwise/BatchNorm/ReadVariableOpReadVariableOpRtrain_mobilenetv1_mobilenetv1_conv2d_7_depthwise_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02K
Itrain/MobilenetV1/MobilenetV1/Conv2d_7_depthwise/BatchNorm/ReadVariableOp?
Ktrain/MobilenetV1/MobilenetV1/Conv2d_7_depthwise/BatchNorm/ReadVariableOp_1ReadVariableOpTtrain_mobilenetv1_mobilenetv1_conv2d_7_depthwise_batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype02M
Ktrain/MobilenetV1/MobilenetV1/Conv2d_7_depthwise/BatchNorm/ReadVariableOp_1?
Ztrain/MobilenetV1/MobilenetV1/Conv2d_7_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOpReadVariableOpctrain_mobilenetv1_mobilenetv1_conv2d_7_depthwise_batchnorm_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02\
Ztrain/MobilenetV1/MobilenetV1/Conv2d_7_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp?
\train/MobilenetV1/MobilenetV1/Conv2d_7_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpetrain_mobilenetv1_mobilenetv1_conv2d_7_depthwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02^
\train/MobilenetV1/MobilenetV1/Conv2d_7_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1?
Ktrain/MobilenetV1/MobilenetV1/Conv2d_7_depthwise/BatchNorm/FusedBatchNormV3FusedBatchNormV3Ctrain/MobilenetV1/MobilenetV1/Conv2d_7_depthwise/depthwise:output:0Qtrain/MobilenetV1/MobilenetV1/Conv2d_7_depthwise/BatchNorm/ReadVariableOp:value:0Strain/MobilenetV1/MobilenetV1/Conv2d_7_depthwise/BatchNorm/ReadVariableOp_1:value:0btrain/MobilenetV1/MobilenetV1/Conv2d_7_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp:value:0dtrain/MobilenetV1/MobilenetV1/Conv2d_7_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:2M
Ktrain/MobilenetV1/MobilenetV1/Conv2d_7_depthwise/BatchNorm/FusedBatchNormV3?
Ptrain/MobilenetV1/MobilenetV1/Conv2d_7_depthwise/BatchNorm/AssignMovingAvg/sub_1Subatrain/MobilenetV1/MobilenetV1/Conv2d_7_depthwise/BatchNorm/AssignMovingAvg/ReadVariableOp:value:0Xtrain/MobilenetV1/MobilenetV1/Conv2d_7_depthwise/BatchNorm/FusedBatchNormV3:batch_mean:0*
T0*
_output_shapes	
:?2R
Ptrain/MobilenetV1/MobilenetV1/Conv2d_7_depthwise/BatchNorm/AssignMovingAvg/sub_1?
Ptrain/MobilenetV1/MobilenetV1/Conv2d_7_depthwise/BatchNorm/AssignMovingAvg/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2R
Ptrain/MobilenetV1/MobilenetV1/Conv2d_7_depthwise/BatchNorm/AssignMovingAvg/sub/x?
Ntrain/MobilenetV1/MobilenetV1/Conv2d_7_depthwise/BatchNorm/AssignMovingAvg/subSubYtrain/MobilenetV1/MobilenetV1/Conv2d_7_depthwise/BatchNorm/AssignMovingAvg/sub/x:output:0placeholder_2*
T0*
_output_shapes
: 2P
Ntrain/MobilenetV1/MobilenetV1/Conv2d_7_depthwise/BatchNorm/AssignMovingAvg/sub?
Ntrain/MobilenetV1/MobilenetV1/Conv2d_7_depthwise/BatchNorm/AssignMovingAvg/mulMulTtrain/MobilenetV1/MobilenetV1/Conv2d_7_depthwise/BatchNorm/AssignMovingAvg/sub_1:z:0Rtrain/MobilenetV1/MobilenetV1/Conv2d_7_depthwise/BatchNorm/AssignMovingAvg/sub:z:0*
T0*
_output_shapes	
:?2P
Ntrain/MobilenetV1/MobilenetV1/Conv2d_7_depthwise/BatchNorm/AssignMovingAvg/mul?
Jtrain/MobilenetV1/MobilenetV1/Conv2d_7_depthwise/BatchNorm/AssignMovingAvgAssignSubVariableOpctrain_mobilenetv1_mobilenetv1_conv2d_7_depthwise_batchnorm_fusedbatchnormv3_readvariableop_resourceRtrain/MobilenetV1/MobilenetV1/Conv2d_7_depthwise/BatchNorm/AssignMovingAvg/mul:z:0*
_output_shapes
 *
dtype02L
Jtrain/MobilenetV1/MobilenetV1/Conv2d_7_depthwise/BatchNorm/AssignMovingAvg?
[train/MobilenetV1/MobilenetV1/Conv2d_7_depthwise/BatchNorm/AssignMovingAvg_1/ReadVariableOpReadVariableOpetrain_mobilenetv1_mobilenetv1_conv2d_7_depthwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02]
[train/MobilenetV1/MobilenetV1/Conv2d_7_depthwise/BatchNorm/AssignMovingAvg_1/ReadVariableOp?
Rtrain/MobilenetV1/MobilenetV1/Conv2d_7_depthwise/BatchNorm/AssignMovingAvg_1/sub_1Subctrain/MobilenetV1/MobilenetV1/Conv2d_7_depthwise/BatchNorm/AssignMovingAvg_1/ReadVariableOp:value:0\train/MobilenetV1/MobilenetV1/Conv2d_7_depthwise/BatchNorm/FusedBatchNormV3:batch_variance:0*
T0*
_output_shapes	
:?2T
Rtrain/MobilenetV1/MobilenetV1/Conv2d_7_depthwise/BatchNorm/AssignMovingAvg_1/sub_1?
Rtrain/MobilenetV1/MobilenetV1/Conv2d_7_depthwise/BatchNorm/AssignMovingAvg_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2T
Rtrain/MobilenetV1/MobilenetV1/Conv2d_7_depthwise/BatchNorm/AssignMovingAvg_1/sub/x?
Ptrain/MobilenetV1/MobilenetV1/Conv2d_7_depthwise/BatchNorm/AssignMovingAvg_1/subSub[train/MobilenetV1/MobilenetV1/Conv2d_7_depthwise/BatchNorm/AssignMovingAvg_1/sub/x:output:0placeholder_2*
T0*
_output_shapes
: 2R
Ptrain/MobilenetV1/MobilenetV1/Conv2d_7_depthwise/BatchNorm/AssignMovingAvg_1/sub?
Ptrain/MobilenetV1/MobilenetV1/Conv2d_7_depthwise/BatchNorm/AssignMovingAvg_1/mulMulVtrain/MobilenetV1/MobilenetV1/Conv2d_7_depthwise/BatchNorm/AssignMovingAvg_1/sub_1:z:0Ttrain/MobilenetV1/MobilenetV1/Conv2d_7_depthwise/BatchNorm/AssignMovingAvg_1/sub:z:0*
T0*
_output_shapes	
:?2R
Ptrain/MobilenetV1/MobilenetV1/Conv2d_7_depthwise/BatchNorm/AssignMovingAvg_1/mul?
Ltrain/MobilenetV1/MobilenetV1/Conv2d_7_depthwise/BatchNorm/AssignMovingAvg_1AssignSubVariableOpetrain_mobilenetv1_mobilenetv1_conv2d_7_depthwise_batchnorm_fusedbatchnormv3_readvariableop_1_resourceTtrain/MobilenetV1/MobilenetV1/Conv2d_7_depthwise/BatchNorm/AssignMovingAvg_1/mul:z:0*
_output_shapes
 *
dtype02N
Ltrain/MobilenetV1/MobilenetV1/Conv2d_7_depthwise/BatchNorm/AssignMovingAvg_1?
Ytrain/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/BatchNorm/AssignMovingAvg/ReadVariableOpReadVariableOpctrain_mobilenetv1_mobilenetv1_conv2d_7_pointwise_batchnorm_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02[
Ytrain/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/BatchNorm/AssignMovingAvg/ReadVariableOp?
6train/MobilenetV1/MobilenetV1/Conv2d_7_depthwise/Relu6Relu6Otrain/MobilenetV1/MobilenetV1/Conv2d_7_depthwise/BatchNorm/FusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,????????????????????????????28
6train/MobilenetV1/MobilenetV1/Conv2d_7_depthwise/Relu6?
Ftrain/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/Conv2D/ReadVariableOpReadVariableOpqtrain_mobilenetv1_mobilenetv1_conv2d_7_pointwise_kernel_regularizer_l2_regularizer_l2loss_readvariableop_resource*(
_output_shapes
:??*
dtype02H
Ftrain/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/Conv2D/ReadVariableOp?
7train/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/Conv2DConv2DDtrain/MobilenetV1/MobilenetV1/Conv2d_7_depthwise/Relu6:activations:0Ntrain/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
29
7train/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/Conv2D?
Itrain/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/BatchNorm/ReadVariableOpReadVariableOpRtrain_mobilenetv1_mobilenetv1_conv2d_7_pointwise_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02K
Itrain/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/BatchNorm/ReadVariableOp?
Ktrain/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/BatchNorm/ReadVariableOp_1ReadVariableOpTtrain_mobilenetv1_mobilenetv1_conv2d_7_pointwise_batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype02M
Ktrain/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/BatchNorm/ReadVariableOp_1?
Ztrain/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOpReadVariableOpctrain_mobilenetv1_mobilenetv1_conv2d_7_pointwise_batchnorm_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02\
Ztrain/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp?
\train/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpetrain_mobilenetv1_mobilenetv1_conv2d_7_pointwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02^
\train/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1?
Ktrain/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/BatchNorm/FusedBatchNormV3FusedBatchNormV3@train/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/Conv2D:output:0Qtrain/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/BatchNorm/ReadVariableOp:value:0Strain/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/BatchNorm/ReadVariableOp_1:value:0btrain/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp:value:0dtrain/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:2M
Ktrain/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/BatchNorm/FusedBatchNormV3?
Ptrain/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/BatchNorm/AssignMovingAvg/sub_1Subatrain/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/BatchNorm/AssignMovingAvg/ReadVariableOp:value:0Xtrain/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/BatchNorm/FusedBatchNormV3:batch_mean:0*
T0*
_output_shapes	
:?2R
Ptrain/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/BatchNorm/AssignMovingAvg/sub_1?
Ptrain/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/BatchNorm/AssignMovingAvg/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2R
Ptrain/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/BatchNorm/AssignMovingAvg/sub/x?
Ntrain/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/BatchNorm/AssignMovingAvg/subSubYtrain/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/BatchNorm/AssignMovingAvg/sub/x:output:0placeholder_2*
T0*
_output_shapes
: 2P
Ntrain/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/BatchNorm/AssignMovingAvg/sub?
Ntrain/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/BatchNorm/AssignMovingAvg/mulMulTtrain/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/BatchNorm/AssignMovingAvg/sub_1:z:0Rtrain/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/BatchNorm/AssignMovingAvg/sub:z:0*
T0*
_output_shapes	
:?2P
Ntrain/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/BatchNorm/AssignMovingAvg/mul?
Jtrain/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/BatchNorm/AssignMovingAvgAssignSubVariableOpctrain_mobilenetv1_mobilenetv1_conv2d_7_pointwise_batchnorm_fusedbatchnormv3_readvariableop_resourceRtrain/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/BatchNorm/AssignMovingAvg/mul:z:0*
_output_shapes
 *
dtype02L
Jtrain/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/BatchNorm/AssignMovingAvg?
[train/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/BatchNorm/AssignMovingAvg_1/ReadVariableOpReadVariableOpetrain_mobilenetv1_mobilenetv1_conv2d_7_pointwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02]
[train/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/BatchNorm/AssignMovingAvg_1/ReadVariableOp?
Rtrain/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/BatchNorm/AssignMovingAvg_1/sub_1Subctrain/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/BatchNorm/AssignMovingAvg_1/ReadVariableOp:value:0\train/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/BatchNorm/FusedBatchNormV3:batch_variance:0*
T0*
_output_shapes	
:?2T
Rtrain/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/BatchNorm/AssignMovingAvg_1/sub_1?
Rtrain/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/BatchNorm/AssignMovingAvg_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2T
Rtrain/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/BatchNorm/AssignMovingAvg_1/sub/x?
Ptrain/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/BatchNorm/AssignMovingAvg_1/subSub[train/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/BatchNorm/AssignMovingAvg_1/sub/x:output:0placeholder_2*
T0*
_output_shapes
: 2R
Ptrain/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/BatchNorm/AssignMovingAvg_1/sub?
Ptrain/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/BatchNorm/AssignMovingAvg_1/mulMulVtrain/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/BatchNorm/AssignMovingAvg_1/sub_1:z:0Ttrain/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/BatchNorm/AssignMovingAvg_1/sub:z:0*
T0*
_output_shapes	
:?2R
Ptrain/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/BatchNorm/AssignMovingAvg_1/mul?
Ltrain/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/BatchNorm/AssignMovingAvg_1AssignSubVariableOpetrain_mobilenetv1_mobilenetv1_conv2d_7_pointwise_batchnorm_fusedbatchnormv3_readvariableop_1_resourceTtrain/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/BatchNorm/AssignMovingAvg_1/mul:z:0*
_output_shapes
 *
dtype02N
Ltrain/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/BatchNorm/AssignMovingAvg_1?
Ytrain/MobilenetV1/MobilenetV1/Conv2d_8_depthwise/BatchNorm/AssignMovingAvg/ReadVariableOpReadVariableOpctrain_mobilenetv1_mobilenetv1_conv2d_8_depthwise_batchnorm_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02[
Ytrain/MobilenetV1/MobilenetV1/Conv2d_8_depthwise/BatchNorm/AssignMovingAvg/ReadVariableOp?
6train/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/Relu6Relu6Otrain/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/BatchNorm/FusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,????????????????????????????28
6train/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/Relu6?
Itrain/MobilenetV1/MobilenetV1/Conv2d_8_depthwise/depthwise/ReadVariableOpReadVariableOpRtrain_mobilenetv1_mobilenetv1_conv2d_8_depthwise_depthwise_readvariableop_resource*'
_output_shapes
:?*
dtype02K
Itrain/MobilenetV1/MobilenetV1/Conv2d_8_depthwise/depthwise/ReadVariableOp?
:train/MobilenetV1/MobilenetV1/Conv2d_8_depthwise/depthwiseDepthwiseConv2dNativeDtrain/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/Relu6:activations:0Qtrain/MobilenetV1/MobilenetV1/Conv2d_8_depthwise/depthwise/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
2<
:train/MobilenetV1/MobilenetV1/Conv2d_8_depthwise/depthwise?
Itrain/MobilenetV1/MobilenetV1/Conv2d_8_depthwise/BatchNorm/ReadVariableOpReadVariableOpRtrain_mobilenetv1_mobilenetv1_conv2d_8_depthwise_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02K
Itrain/MobilenetV1/MobilenetV1/Conv2d_8_depthwise/BatchNorm/ReadVariableOp?
Ktrain/MobilenetV1/MobilenetV1/Conv2d_8_depthwise/BatchNorm/ReadVariableOp_1ReadVariableOpTtrain_mobilenetv1_mobilenetv1_conv2d_8_depthwise_batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype02M
Ktrain/MobilenetV1/MobilenetV1/Conv2d_8_depthwise/BatchNorm/ReadVariableOp_1?
Ztrain/MobilenetV1/MobilenetV1/Conv2d_8_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOpReadVariableOpctrain_mobilenetv1_mobilenetv1_conv2d_8_depthwise_batchnorm_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02\
Ztrain/MobilenetV1/MobilenetV1/Conv2d_8_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp?
\train/MobilenetV1/MobilenetV1/Conv2d_8_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpetrain_mobilenetv1_mobilenetv1_conv2d_8_depthwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02^
\train/MobilenetV1/MobilenetV1/Conv2d_8_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1?
Ktrain/MobilenetV1/MobilenetV1/Conv2d_8_depthwise/BatchNorm/FusedBatchNormV3FusedBatchNormV3Ctrain/MobilenetV1/MobilenetV1/Conv2d_8_depthwise/depthwise:output:0Qtrain/MobilenetV1/MobilenetV1/Conv2d_8_depthwise/BatchNorm/ReadVariableOp:value:0Strain/MobilenetV1/MobilenetV1/Conv2d_8_depthwise/BatchNorm/ReadVariableOp_1:value:0btrain/MobilenetV1/MobilenetV1/Conv2d_8_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp:value:0dtrain/MobilenetV1/MobilenetV1/Conv2d_8_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:2M
Ktrain/MobilenetV1/MobilenetV1/Conv2d_8_depthwise/BatchNorm/FusedBatchNormV3?
Ptrain/MobilenetV1/MobilenetV1/Conv2d_8_depthwise/BatchNorm/AssignMovingAvg/sub_1Subatrain/MobilenetV1/MobilenetV1/Conv2d_8_depthwise/BatchNorm/AssignMovingAvg/ReadVariableOp:value:0Xtrain/MobilenetV1/MobilenetV1/Conv2d_8_depthwise/BatchNorm/FusedBatchNormV3:batch_mean:0*
T0*
_output_shapes	
:?2R
Ptrain/MobilenetV1/MobilenetV1/Conv2d_8_depthwise/BatchNorm/AssignMovingAvg/sub_1?
Ptrain/MobilenetV1/MobilenetV1/Conv2d_8_depthwise/BatchNorm/AssignMovingAvg/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2R
Ptrain/MobilenetV1/MobilenetV1/Conv2d_8_depthwise/BatchNorm/AssignMovingAvg/sub/x?
Ntrain/MobilenetV1/MobilenetV1/Conv2d_8_depthwise/BatchNorm/AssignMovingAvg/subSubYtrain/MobilenetV1/MobilenetV1/Conv2d_8_depthwise/BatchNorm/AssignMovingAvg/sub/x:output:0placeholder_2*
T0*
_output_shapes
: 2P
Ntrain/MobilenetV1/MobilenetV1/Conv2d_8_depthwise/BatchNorm/AssignMovingAvg/sub?
Ntrain/MobilenetV1/MobilenetV1/Conv2d_8_depthwise/BatchNorm/AssignMovingAvg/mulMulTtrain/MobilenetV1/MobilenetV1/Conv2d_8_depthwise/BatchNorm/AssignMovingAvg/sub_1:z:0Rtrain/MobilenetV1/MobilenetV1/Conv2d_8_depthwise/BatchNorm/AssignMovingAvg/sub:z:0*
T0*
_output_shapes	
:?2P
Ntrain/MobilenetV1/MobilenetV1/Conv2d_8_depthwise/BatchNorm/AssignMovingAvg/mul?
Jtrain/MobilenetV1/MobilenetV1/Conv2d_8_depthwise/BatchNorm/AssignMovingAvgAssignSubVariableOpctrain_mobilenetv1_mobilenetv1_conv2d_8_depthwise_batchnorm_fusedbatchnormv3_readvariableop_resourceRtrain/MobilenetV1/MobilenetV1/Conv2d_8_depthwise/BatchNorm/AssignMovingAvg/mul:z:0*
_output_shapes
 *
dtype02L
Jtrain/MobilenetV1/MobilenetV1/Conv2d_8_depthwise/BatchNorm/AssignMovingAvg?
[train/MobilenetV1/MobilenetV1/Conv2d_8_depthwise/BatchNorm/AssignMovingAvg_1/ReadVariableOpReadVariableOpetrain_mobilenetv1_mobilenetv1_conv2d_8_depthwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02]
[train/MobilenetV1/MobilenetV1/Conv2d_8_depthwise/BatchNorm/AssignMovingAvg_1/ReadVariableOp?
Rtrain/MobilenetV1/MobilenetV1/Conv2d_8_depthwise/BatchNorm/AssignMovingAvg_1/sub_1Subctrain/MobilenetV1/MobilenetV1/Conv2d_8_depthwise/BatchNorm/AssignMovingAvg_1/ReadVariableOp:value:0\train/MobilenetV1/MobilenetV1/Conv2d_8_depthwise/BatchNorm/FusedBatchNormV3:batch_variance:0*
T0*
_output_shapes	
:?2T
Rtrain/MobilenetV1/MobilenetV1/Conv2d_8_depthwise/BatchNorm/AssignMovingAvg_1/sub_1?
Rtrain/MobilenetV1/MobilenetV1/Conv2d_8_depthwise/BatchNorm/AssignMovingAvg_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2T
Rtrain/MobilenetV1/MobilenetV1/Conv2d_8_depthwise/BatchNorm/AssignMovingAvg_1/sub/x?
Ptrain/MobilenetV1/MobilenetV1/Conv2d_8_depthwise/BatchNorm/AssignMovingAvg_1/subSub[train/MobilenetV1/MobilenetV1/Conv2d_8_depthwise/BatchNorm/AssignMovingAvg_1/sub/x:output:0placeholder_2*
T0*
_output_shapes
: 2R
Ptrain/MobilenetV1/MobilenetV1/Conv2d_8_depthwise/BatchNorm/AssignMovingAvg_1/sub?
Ptrain/MobilenetV1/MobilenetV1/Conv2d_8_depthwise/BatchNorm/AssignMovingAvg_1/mulMulVtrain/MobilenetV1/MobilenetV1/Conv2d_8_depthwise/BatchNorm/AssignMovingAvg_1/sub_1:z:0Ttrain/MobilenetV1/MobilenetV1/Conv2d_8_depthwise/BatchNorm/AssignMovingAvg_1/sub:z:0*
T0*
_output_shapes	
:?2R
Ptrain/MobilenetV1/MobilenetV1/Conv2d_8_depthwise/BatchNorm/AssignMovingAvg_1/mul?
Ltrain/MobilenetV1/MobilenetV1/Conv2d_8_depthwise/BatchNorm/AssignMovingAvg_1AssignSubVariableOpetrain_mobilenetv1_mobilenetv1_conv2d_8_depthwise_batchnorm_fusedbatchnormv3_readvariableop_1_resourceTtrain/MobilenetV1/MobilenetV1/Conv2d_8_depthwise/BatchNorm/AssignMovingAvg_1/mul:z:0*
_output_shapes
 *
dtype02N
Ltrain/MobilenetV1/MobilenetV1/Conv2d_8_depthwise/BatchNorm/AssignMovingAvg_1?
Ytrain/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/BatchNorm/AssignMovingAvg/ReadVariableOpReadVariableOpctrain_mobilenetv1_mobilenetv1_conv2d_8_pointwise_batchnorm_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02[
Ytrain/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/BatchNorm/AssignMovingAvg/ReadVariableOp?
6train/MobilenetV1/MobilenetV1/Conv2d_8_depthwise/Relu6Relu6Otrain/MobilenetV1/MobilenetV1/Conv2d_8_depthwise/BatchNorm/FusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,????????????????????????????28
6train/MobilenetV1/MobilenetV1/Conv2d_8_depthwise/Relu6?
Ftrain/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/Conv2D/ReadVariableOpReadVariableOpqtrain_mobilenetv1_mobilenetv1_conv2d_8_pointwise_kernel_regularizer_l2_regularizer_l2loss_readvariableop_resource*(
_output_shapes
:??*
dtype02H
Ftrain/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/Conv2D/ReadVariableOp?
7train/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/Conv2DConv2DDtrain/MobilenetV1/MobilenetV1/Conv2d_8_depthwise/Relu6:activations:0Ntrain/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
29
7train/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/Conv2D?
Itrain/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/BatchNorm/ReadVariableOpReadVariableOpRtrain_mobilenetv1_mobilenetv1_conv2d_8_pointwise_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02K
Itrain/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/BatchNorm/ReadVariableOp?
Ktrain/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/BatchNorm/ReadVariableOp_1ReadVariableOpTtrain_mobilenetv1_mobilenetv1_conv2d_8_pointwise_batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype02M
Ktrain/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/BatchNorm/ReadVariableOp_1?
Ztrain/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOpReadVariableOpctrain_mobilenetv1_mobilenetv1_conv2d_8_pointwise_batchnorm_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02\
Ztrain/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp?
\train/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpetrain_mobilenetv1_mobilenetv1_conv2d_8_pointwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02^
\train/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1?
Ktrain/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/BatchNorm/FusedBatchNormV3FusedBatchNormV3@train/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/Conv2D:output:0Qtrain/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/BatchNorm/ReadVariableOp:value:0Strain/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/BatchNorm/ReadVariableOp_1:value:0btrain/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp:value:0dtrain/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:2M
Ktrain/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/BatchNorm/FusedBatchNormV3?
Ptrain/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/BatchNorm/AssignMovingAvg/sub_1Subatrain/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/BatchNorm/AssignMovingAvg/ReadVariableOp:value:0Xtrain/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/BatchNorm/FusedBatchNormV3:batch_mean:0*
T0*
_output_shapes	
:?2R
Ptrain/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/BatchNorm/AssignMovingAvg/sub_1?
Ptrain/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/BatchNorm/AssignMovingAvg/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2R
Ptrain/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/BatchNorm/AssignMovingAvg/sub/x?
Ntrain/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/BatchNorm/AssignMovingAvg/subSubYtrain/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/BatchNorm/AssignMovingAvg/sub/x:output:0placeholder_2*
T0*
_output_shapes
: 2P
Ntrain/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/BatchNorm/AssignMovingAvg/sub?
Ntrain/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/BatchNorm/AssignMovingAvg/mulMulTtrain/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/BatchNorm/AssignMovingAvg/sub_1:z:0Rtrain/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/BatchNorm/AssignMovingAvg/sub:z:0*
T0*
_output_shapes	
:?2P
Ntrain/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/BatchNorm/AssignMovingAvg/mul?
Jtrain/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/BatchNorm/AssignMovingAvgAssignSubVariableOpctrain_mobilenetv1_mobilenetv1_conv2d_8_pointwise_batchnorm_fusedbatchnormv3_readvariableop_resourceRtrain/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/BatchNorm/AssignMovingAvg/mul:z:0*
_output_shapes
 *
dtype02L
Jtrain/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/BatchNorm/AssignMovingAvg?
[train/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/BatchNorm/AssignMovingAvg_1/ReadVariableOpReadVariableOpetrain_mobilenetv1_mobilenetv1_conv2d_8_pointwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02]
[train/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/BatchNorm/AssignMovingAvg_1/ReadVariableOp?
Rtrain/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/BatchNorm/AssignMovingAvg_1/sub_1Subctrain/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/BatchNorm/AssignMovingAvg_1/ReadVariableOp:value:0\train/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/BatchNorm/FusedBatchNormV3:batch_variance:0*
T0*
_output_shapes	
:?2T
Rtrain/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/BatchNorm/AssignMovingAvg_1/sub_1?
Rtrain/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/BatchNorm/AssignMovingAvg_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2T
Rtrain/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/BatchNorm/AssignMovingAvg_1/sub/x?
Ptrain/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/BatchNorm/AssignMovingAvg_1/subSub[train/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/BatchNorm/AssignMovingAvg_1/sub/x:output:0placeholder_2*
T0*
_output_shapes
: 2R
Ptrain/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/BatchNorm/AssignMovingAvg_1/sub?
Ptrain/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/BatchNorm/AssignMovingAvg_1/mulMulVtrain/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/BatchNorm/AssignMovingAvg_1/sub_1:z:0Ttrain/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/BatchNorm/AssignMovingAvg_1/sub:z:0*
T0*
_output_shapes	
:?2R
Ptrain/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/BatchNorm/AssignMovingAvg_1/mul?
Ltrain/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/BatchNorm/AssignMovingAvg_1AssignSubVariableOpetrain_mobilenetv1_mobilenetv1_conv2d_8_pointwise_batchnorm_fusedbatchnormv3_readvariableop_1_resourceTtrain/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/BatchNorm/AssignMovingAvg_1/mul:z:0*
_output_shapes
 *
dtype02N
Ltrain/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/BatchNorm/AssignMovingAvg_1?
Ytrain/MobilenetV1/MobilenetV1/Conv2d_9_depthwise/BatchNorm/AssignMovingAvg/ReadVariableOpReadVariableOpctrain_mobilenetv1_mobilenetv1_conv2d_9_depthwise_batchnorm_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02[
Ytrain/MobilenetV1/MobilenetV1/Conv2d_9_depthwise/BatchNorm/AssignMovingAvg/ReadVariableOp?
6train/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/Relu6Relu6Otrain/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/BatchNorm/FusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,????????????????????????????28
6train/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/Relu6?
Itrain/MobilenetV1/MobilenetV1/Conv2d_9_depthwise/depthwise/ReadVariableOpReadVariableOpRtrain_mobilenetv1_mobilenetv1_conv2d_9_depthwise_depthwise_readvariableop_resource*'
_output_shapes
:?*
dtype02K
Itrain/MobilenetV1/MobilenetV1/Conv2d_9_depthwise/depthwise/ReadVariableOp?
:train/MobilenetV1/MobilenetV1/Conv2d_9_depthwise/depthwiseDepthwiseConv2dNativeDtrain/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/Relu6:activations:0Qtrain/MobilenetV1/MobilenetV1/Conv2d_9_depthwise/depthwise/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
2<
:train/MobilenetV1/MobilenetV1/Conv2d_9_depthwise/depthwise?
Itrain/MobilenetV1/MobilenetV1/Conv2d_9_depthwise/BatchNorm/ReadVariableOpReadVariableOpRtrain_mobilenetv1_mobilenetv1_conv2d_9_depthwise_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02K
Itrain/MobilenetV1/MobilenetV1/Conv2d_9_depthwise/BatchNorm/ReadVariableOp?
Ktrain/MobilenetV1/MobilenetV1/Conv2d_9_depthwise/BatchNorm/ReadVariableOp_1ReadVariableOpTtrain_mobilenetv1_mobilenetv1_conv2d_9_depthwise_batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype02M
Ktrain/MobilenetV1/MobilenetV1/Conv2d_9_depthwise/BatchNorm/ReadVariableOp_1?
Ztrain/MobilenetV1/MobilenetV1/Conv2d_9_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOpReadVariableOpctrain_mobilenetv1_mobilenetv1_conv2d_9_depthwise_batchnorm_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02\
Ztrain/MobilenetV1/MobilenetV1/Conv2d_9_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp?
\train/MobilenetV1/MobilenetV1/Conv2d_9_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpetrain_mobilenetv1_mobilenetv1_conv2d_9_depthwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02^
\train/MobilenetV1/MobilenetV1/Conv2d_9_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1?
Ktrain/MobilenetV1/MobilenetV1/Conv2d_9_depthwise/BatchNorm/FusedBatchNormV3FusedBatchNormV3Ctrain/MobilenetV1/MobilenetV1/Conv2d_9_depthwise/depthwise:output:0Qtrain/MobilenetV1/MobilenetV1/Conv2d_9_depthwise/BatchNorm/ReadVariableOp:value:0Strain/MobilenetV1/MobilenetV1/Conv2d_9_depthwise/BatchNorm/ReadVariableOp_1:value:0btrain/MobilenetV1/MobilenetV1/Conv2d_9_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp:value:0dtrain/MobilenetV1/MobilenetV1/Conv2d_9_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:2M
Ktrain/MobilenetV1/MobilenetV1/Conv2d_9_depthwise/BatchNorm/FusedBatchNormV3?
Ptrain/MobilenetV1/MobilenetV1/Conv2d_9_depthwise/BatchNorm/AssignMovingAvg/sub_1Subatrain/MobilenetV1/MobilenetV1/Conv2d_9_depthwise/BatchNorm/AssignMovingAvg/ReadVariableOp:value:0Xtrain/MobilenetV1/MobilenetV1/Conv2d_9_depthwise/BatchNorm/FusedBatchNormV3:batch_mean:0*
T0*
_output_shapes	
:?2R
Ptrain/MobilenetV1/MobilenetV1/Conv2d_9_depthwise/BatchNorm/AssignMovingAvg/sub_1?
Ptrain/MobilenetV1/MobilenetV1/Conv2d_9_depthwise/BatchNorm/AssignMovingAvg/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2R
Ptrain/MobilenetV1/MobilenetV1/Conv2d_9_depthwise/BatchNorm/AssignMovingAvg/sub/x?
Ntrain/MobilenetV1/MobilenetV1/Conv2d_9_depthwise/BatchNorm/AssignMovingAvg/subSubYtrain/MobilenetV1/MobilenetV1/Conv2d_9_depthwise/BatchNorm/AssignMovingAvg/sub/x:output:0placeholder_2*
T0*
_output_shapes
: 2P
Ntrain/MobilenetV1/MobilenetV1/Conv2d_9_depthwise/BatchNorm/AssignMovingAvg/sub?
Ntrain/MobilenetV1/MobilenetV1/Conv2d_9_depthwise/BatchNorm/AssignMovingAvg/mulMulTtrain/MobilenetV1/MobilenetV1/Conv2d_9_depthwise/BatchNorm/AssignMovingAvg/sub_1:z:0Rtrain/MobilenetV1/MobilenetV1/Conv2d_9_depthwise/BatchNorm/AssignMovingAvg/sub:z:0*
T0*
_output_shapes	
:?2P
Ntrain/MobilenetV1/MobilenetV1/Conv2d_9_depthwise/BatchNorm/AssignMovingAvg/mul?
Jtrain/MobilenetV1/MobilenetV1/Conv2d_9_depthwise/BatchNorm/AssignMovingAvgAssignSubVariableOpctrain_mobilenetv1_mobilenetv1_conv2d_9_depthwise_batchnorm_fusedbatchnormv3_readvariableop_resourceRtrain/MobilenetV1/MobilenetV1/Conv2d_9_depthwise/BatchNorm/AssignMovingAvg/mul:z:0*
_output_shapes
 *
dtype02L
Jtrain/MobilenetV1/MobilenetV1/Conv2d_9_depthwise/BatchNorm/AssignMovingAvg?
[train/MobilenetV1/MobilenetV1/Conv2d_9_depthwise/BatchNorm/AssignMovingAvg_1/ReadVariableOpReadVariableOpetrain_mobilenetv1_mobilenetv1_conv2d_9_depthwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02]
[train/MobilenetV1/MobilenetV1/Conv2d_9_depthwise/BatchNorm/AssignMovingAvg_1/ReadVariableOp?
Rtrain/MobilenetV1/MobilenetV1/Conv2d_9_depthwise/BatchNorm/AssignMovingAvg_1/sub_1Subctrain/MobilenetV1/MobilenetV1/Conv2d_9_depthwise/BatchNorm/AssignMovingAvg_1/ReadVariableOp:value:0\train/MobilenetV1/MobilenetV1/Conv2d_9_depthwise/BatchNorm/FusedBatchNormV3:batch_variance:0*
T0*
_output_shapes	
:?2T
Rtrain/MobilenetV1/MobilenetV1/Conv2d_9_depthwise/BatchNorm/AssignMovingAvg_1/sub_1?
Rtrain/MobilenetV1/MobilenetV1/Conv2d_9_depthwise/BatchNorm/AssignMovingAvg_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2T
Rtrain/MobilenetV1/MobilenetV1/Conv2d_9_depthwise/BatchNorm/AssignMovingAvg_1/sub/x?
Ptrain/MobilenetV1/MobilenetV1/Conv2d_9_depthwise/BatchNorm/AssignMovingAvg_1/subSub[train/MobilenetV1/MobilenetV1/Conv2d_9_depthwise/BatchNorm/AssignMovingAvg_1/sub/x:output:0placeholder_2*
T0*
_output_shapes
: 2R
Ptrain/MobilenetV1/MobilenetV1/Conv2d_9_depthwise/BatchNorm/AssignMovingAvg_1/sub?
Ptrain/MobilenetV1/MobilenetV1/Conv2d_9_depthwise/BatchNorm/AssignMovingAvg_1/mulMulVtrain/MobilenetV1/MobilenetV1/Conv2d_9_depthwise/BatchNorm/AssignMovingAvg_1/sub_1:z:0Ttrain/MobilenetV1/MobilenetV1/Conv2d_9_depthwise/BatchNorm/AssignMovingAvg_1/sub:z:0*
T0*
_output_shapes	
:?2R
Ptrain/MobilenetV1/MobilenetV1/Conv2d_9_depthwise/BatchNorm/AssignMovingAvg_1/mul?
Ltrain/MobilenetV1/MobilenetV1/Conv2d_9_depthwise/BatchNorm/AssignMovingAvg_1AssignSubVariableOpetrain_mobilenetv1_mobilenetv1_conv2d_9_depthwise_batchnorm_fusedbatchnormv3_readvariableop_1_resourceTtrain/MobilenetV1/MobilenetV1/Conv2d_9_depthwise/BatchNorm/AssignMovingAvg_1/mul:z:0*
_output_shapes
 *
dtype02N
Ltrain/MobilenetV1/MobilenetV1/Conv2d_9_depthwise/BatchNorm/AssignMovingAvg_1?
Ytrain/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/BatchNorm/AssignMovingAvg/ReadVariableOpReadVariableOpctrain_mobilenetv1_mobilenetv1_conv2d_9_pointwise_batchnorm_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02[
Ytrain/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/BatchNorm/AssignMovingAvg/ReadVariableOp?
6train/MobilenetV1/MobilenetV1/Conv2d_9_depthwise/Relu6Relu6Otrain/MobilenetV1/MobilenetV1/Conv2d_9_depthwise/BatchNorm/FusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,????????????????????????????28
6train/MobilenetV1/MobilenetV1/Conv2d_9_depthwise/Relu6?
Ftrain/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/Conv2D/ReadVariableOpReadVariableOpqtrain_mobilenetv1_mobilenetv1_conv2d_9_pointwise_kernel_regularizer_l2_regularizer_l2loss_readvariableop_resource*(
_output_shapes
:??*
dtype02H
Ftrain/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/Conv2D/ReadVariableOp?
7train/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/Conv2DConv2DDtrain/MobilenetV1/MobilenetV1/Conv2d_9_depthwise/Relu6:activations:0Ntrain/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
29
7train/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/Conv2D?
Itrain/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/BatchNorm/ReadVariableOpReadVariableOpRtrain_mobilenetv1_mobilenetv1_conv2d_9_pointwise_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02K
Itrain/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/BatchNorm/ReadVariableOp?
Ktrain/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/BatchNorm/ReadVariableOp_1ReadVariableOpTtrain_mobilenetv1_mobilenetv1_conv2d_9_pointwise_batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype02M
Ktrain/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/BatchNorm/ReadVariableOp_1?
Ztrain/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOpReadVariableOpctrain_mobilenetv1_mobilenetv1_conv2d_9_pointwise_batchnorm_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02\
Ztrain/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp?
\train/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpetrain_mobilenetv1_mobilenetv1_conv2d_9_pointwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02^
\train/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1?
Ktrain/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/BatchNorm/FusedBatchNormV3FusedBatchNormV3@train/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/Conv2D:output:0Qtrain/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/BatchNorm/ReadVariableOp:value:0Strain/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/BatchNorm/ReadVariableOp_1:value:0btrain/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp:value:0dtrain/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:2M
Ktrain/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/BatchNorm/FusedBatchNormV3?
Ptrain/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/BatchNorm/AssignMovingAvg/sub_1Subatrain/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/BatchNorm/AssignMovingAvg/ReadVariableOp:value:0Xtrain/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/BatchNorm/FusedBatchNormV3:batch_mean:0*
T0*
_output_shapes	
:?2R
Ptrain/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/BatchNorm/AssignMovingAvg/sub_1?
Ptrain/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/BatchNorm/AssignMovingAvg/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2R
Ptrain/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/BatchNorm/AssignMovingAvg/sub/x?
Ntrain/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/BatchNorm/AssignMovingAvg/subSubYtrain/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/BatchNorm/AssignMovingAvg/sub/x:output:0placeholder_2*
T0*
_output_shapes
: 2P
Ntrain/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/BatchNorm/AssignMovingAvg/sub?
Ntrain/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/BatchNorm/AssignMovingAvg/mulMulTtrain/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/BatchNorm/AssignMovingAvg/sub_1:z:0Rtrain/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/BatchNorm/AssignMovingAvg/sub:z:0*
T0*
_output_shapes	
:?2P
Ntrain/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/BatchNorm/AssignMovingAvg/mul?
Jtrain/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/BatchNorm/AssignMovingAvgAssignSubVariableOpctrain_mobilenetv1_mobilenetv1_conv2d_9_pointwise_batchnorm_fusedbatchnormv3_readvariableop_resourceRtrain/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/BatchNorm/AssignMovingAvg/mul:z:0*
_output_shapes
 *
dtype02L
Jtrain/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/BatchNorm/AssignMovingAvg?
[train/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/BatchNorm/AssignMovingAvg_1/ReadVariableOpReadVariableOpetrain_mobilenetv1_mobilenetv1_conv2d_9_pointwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02]
[train/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/BatchNorm/AssignMovingAvg_1/ReadVariableOp?
Rtrain/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/BatchNorm/AssignMovingAvg_1/sub_1Subctrain/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/BatchNorm/AssignMovingAvg_1/ReadVariableOp:value:0\train/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/BatchNorm/FusedBatchNormV3:batch_variance:0*
T0*
_output_shapes	
:?2T
Rtrain/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/BatchNorm/AssignMovingAvg_1/sub_1?
Rtrain/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/BatchNorm/AssignMovingAvg_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2T
Rtrain/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/BatchNorm/AssignMovingAvg_1/sub/x?
Ptrain/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/BatchNorm/AssignMovingAvg_1/subSub[train/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/BatchNorm/AssignMovingAvg_1/sub/x:output:0placeholder_2*
T0*
_output_shapes
: 2R
Ptrain/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/BatchNorm/AssignMovingAvg_1/sub?
Ptrain/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/BatchNorm/AssignMovingAvg_1/mulMulVtrain/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/BatchNorm/AssignMovingAvg_1/sub_1:z:0Ttrain/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/BatchNorm/AssignMovingAvg_1/sub:z:0*
T0*
_output_shapes	
:?2R
Ptrain/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/BatchNorm/AssignMovingAvg_1/mul?
Ltrain/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/BatchNorm/AssignMovingAvg_1AssignSubVariableOpetrain_mobilenetv1_mobilenetv1_conv2d_9_pointwise_batchnorm_fusedbatchnormv3_readvariableop_1_resourceTtrain/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/BatchNorm/AssignMovingAvg_1/mul:z:0*
_output_shapes
 *
dtype02N
Ltrain/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/BatchNorm/AssignMovingAvg_1?
Ztrain/MobilenetV1/MobilenetV1/Conv2d_10_depthwise/BatchNorm/AssignMovingAvg/ReadVariableOpReadVariableOpdtrain_mobilenetv1_mobilenetv1_conv2d_10_depthwise_batchnorm_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02\
Ztrain/MobilenetV1/MobilenetV1/Conv2d_10_depthwise/BatchNorm/AssignMovingAvg/ReadVariableOp?
6train/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/Relu6Relu6Otrain/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/BatchNorm/FusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,????????????????????????????28
6train/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/Relu6?
Jtrain/MobilenetV1/MobilenetV1/Conv2d_10_depthwise/depthwise/ReadVariableOpReadVariableOpStrain_mobilenetv1_mobilenetv1_conv2d_10_depthwise_depthwise_readvariableop_resource*'
_output_shapes
:?*
dtype02L
Jtrain/MobilenetV1/MobilenetV1/Conv2d_10_depthwise/depthwise/ReadVariableOp?
;train/MobilenetV1/MobilenetV1/Conv2d_10_depthwise/depthwiseDepthwiseConv2dNativeDtrain/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/Relu6:activations:0Rtrain/MobilenetV1/MobilenetV1/Conv2d_10_depthwise/depthwise/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
2=
;train/MobilenetV1/MobilenetV1/Conv2d_10_depthwise/depthwise?
Jtrain/MobilenetV1/MobilenetV1/Conv2d_10_depthwise/BatchNorm/ReadVariableOpReadVariableOpStrain_mobilenetv1_mobilenetv1_conv2d_10_depthwise_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02L
Jtrain/MobilenetV1/MobilenetV1/Conv2d_10_depthwise/BatchNorm/ReadVariableOp?
Ltrain/MobilenetV1/MobilenetV1/Conv2d_10_depthwise/BatchNorm/ReadVariableOp_1ReadVariableOpUtrain_mobilenetv1_mobilenetv1_conv2d_10_depthwise_batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype02N
Ltrain/MobilenetV1/MobilenetV1/Conv2d_10_depthwise/BatchNorm/ReadVariableOp_1?
[train/MobilenetV1/MobilenetV1/Conv2d_10_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOpReadVariableOpdtrain_mobilenetv1_mobilenetv1_conv2d_10_depthwise_batchnorm_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02]
[train/MobilenetV1/MobilenetV1/Conv2d_10_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp?
]train/MobilenetV1/MobilenetV1/Conv2d_10_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpftrain_mobilenetv1_mobilenetv1_conv2d_10_depthwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02_
]train/MobilenetV1/MobilenetV1/Conv2d_10_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1?
Ltrain/MobilenetV1/MobilenetV1/Conv2d_10_depthwise/BatchNorm/FusedBatchNormV3FusedBatchNormV3Dtrain/MobilenetV1/MobilenetV1/Conv2d_10_depthwise/depthwise:output:0Rtrain/MobilenetV1/MobilenetV1/Conv2d_10_depthwise/BatchNorm/ReadVariableOp:value:0Ttrain/MobilenetV1/MobilenetV1/Conv2d_10_depthwise/BatchNorm/ReadVariableOp_1:value:0ctrain/MobilenetV1/MobilenetV1/Conv2d_10_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp:value:0etrain/MobilenetV1/MobilenetV1/Conv2d_10_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:2N
Ltrain/MobilenetV1/MobilenetV1/Conv2d_10_depthwise/BatchNorm/FusedBatchNormV3?
Qtrain/MobilenetV1/MobilenetV1/Conv2d_10_depthwise/BatchNorm/AssignMovingAvg/sub_1Subbtrain/MobilenetV1/MobilenetV1/Conv2d_10_depthwise/BatchNorm/AssignMovingAvg/ReadVariableOp:value:0Ytrain/MobilenetV1/MobilenetV1/Conv2d_10_depthwise/BatchNorm/FusedBatchNormV3:batch_mean:0*
T0*
_output_shapes	
:?2S
Qtrain/MobilenetV1/MobilenetV1/Conv2d_10_depthwise/BatchNorm/AssignMovingAvg/sub_1?
Qtrain/MobilenetV1/MobilenetV1/Conv2d_10_depthwise/BatchNorm/AssignMovingAvg/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2S
Qtrain/MobilenetV1/MobilenetV1/Conv2d_10_depthwise/BatchNorm/AssignMovingAvg/sub/x?
Otrain/MobilenetV1/MobilenetV1/Conv2d_10_depthwise/BatchNorm/AssignMovingAvg/subSubZtrain/MobilenetV1/MobilenetV1/Conv2d_10_depthwise/BatchNorm/AssignMovingAvg/sub/x:output:0placeholder_2*
T0*
_output_shapes
: 2Q
Otrain/MobilenetV1/MobilenetV1/Conv2d_10_depthwise/BatchNorm/AssignMovingAvg/sub?
Otrain/MobilenetV1/MobilenetV1/Conv2d_10_depthwise/BatchNorm/AssignMovingAvg/mulMulUtrain/MobilenetV1/MobilenetV1/Conv2d_10_depthwise/BatchNorm/AssignMovingAvg/sub_1:z:0Strain/MobilenetV1/MobilenetV1/Conv2d_10_depthwise/BatchNorm/AssignMovingAvg/sub:z:0*
T0*
_output_shapes	
:?2Q
Otrain/MobilenetV1/MobilenetV1/Conv2d_10_depthwise/BatchNorm/AssignMovingAvg/mul?
Ktrain/MobilenetV1/MobilenetV1/Conv2d_10_depthwise/BatchNorm/AssignMovingAvgAssignSubVariableOpdtrain_mobilenetv1_mobilenetv1_conv2d_10_depthwise_batchnorm_fusedbatchnormv3_readvariableop_resourceStrain/MobilenetV1/MobilenetV1/Conv2d_10_depthwise/BatchNorm/AssignMovingAvg/mul:z:0*
_output_shapes
 *
dtype02M
Ktrain/MobilenetV1/MobilenetV1/Conv2d_10_depthwise/BatchNorm/AssignMovingAvg?
\train/MobilenetV1/MobilenetV1/Conv2d_10_depthwise/BatchNorm/AssignMovingAvg_1/ReadVariableOpReadVariableOpftrain_mobilenetv1_mobilenetv1_conv2d_10_depthwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02^
\train/MobilenetV1/MobilenetV1/Conv2d_10_depthwise/BatchNorm/AssignMovingAvg_1/ReadVariableOp?
Strain/MobilenetV1/MobilenetV1/Conv2d_10_depthwise/BatchNorm/AssignMovingAvg_1/sub_1Subdtrain/MobilenetV1/MobilenetV1/Conv2d_10_depthwise/BatchNorm/AssignMovingAvg_1/ReadVariableOp:value:0]train/MobilenetV1/MobilenetV1/Conv2d_10_depthwise/BatchNorm/FusedBatchNormV3:batch_variance:0*
T0*
_output_shapes	
:?2U
Strain/MobilenetV1/MobilenetV1/Conv2d_10_depthwise/BatchNorm/AssignMovingAvg_1/sub_1?
Strain/MobilenetV1/MobilenetV1/Conv2d_10_depthwise/BatchNorm/AssignMovingAvg_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2U
Strain/MobilenetV1/MobilenetV1/Conv2d_10_depthwise/BatchNorm/AssignMovingAvg_1/sub/x?
Qtrain/MobilenetV1/MobilenetV1/Conv2d_10_depthwise/BatchNorm/AssignMovingAvg_1/subSub\train/MobilenetV1/MobilenetV1/Conv2d_10_depthwise/BatchNorm/AssignMovingAvg_1/sub/x:output:0placeholder_2*
T0*
_output_shapes
: 2S
Qtrain/MobilenetV1/MobilenetV1/Conv2d_10_depthwise/BatchNorm/AssignMovingAvg_1/sub?
Qtrain/MobilenetV1/MobilenetV1/Conv2d_10_depthwise/BatchNorm/AssignMovingAvg_1/mulMulWtrain/MobilenetV1/MobilenetV1/Conv2d_10_depthwise/BatchNorm/AssignMovingAvg_1/sub_1:z:0Utrain/MobilenetV1/MobilenetV1/Conv2d_10_depthwise/BatchNorm/AssignMovingAvg_1/sub:z:0*
T0*
_output_shapes	
:?2S
Qtrain/MobilenetV1/MobilenetV1/Conv2d_10_depthwise/BatchNorm/AssignMovingAvg_1/mul?
Mtrain/MobilenetV1/MobilenetV1/Conv2d_10_depthwise/BatchNorm/AssignMovingAvg_1AssignSubVariableOpftrain_mobilenetv1_mobilenetv1_conv2d_10_depthwise_batchnorm_fusedbatchnormv3_readvariableop_1_resourceUtrain/MobilenetV1/MobilenetV1/Conv2d_10_depthwise/BatchNorm/AssignMovingAvg_1/mul:z:0*
_output_shapes
 *
dtype02O
Mtrain/MobilenetV1/MobilenetV1/Conv2d_10_depthwise/BatchNorm/AssignMovingAvg_1?
Ztrain/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/BatchNorm/AssignMovingAvg/ReadVariableOpReadVariableOpdtrain_mobilenetv1_mobilenetv1_conv2d_10_pointwise_batchnorm_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02\
Ztrain/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/BatchNorm/AssignMovingAvg/ReadVariableOp?
7train/MobilenetV1/MobilenetV1/Conv2d_10_depthwise/Relu6Relu6Ptrain/MobilenetV1/MobilenetV1/Conv2d_10_depthwise/BatchNorm/FusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,????????????????????????????29
7train/MobilenetV1/MobilenetV1/Conv2d_10_depthwise/Relu6?
Gtrain/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/Conv2D/ReadVariableOpReadVariableOprtrain_mobilenetv1_mobilenetv1_conv2d_10_pointwise_kernel_regularizer_l2_regularizer_l2loss_readvariableop_resource*(
_output_shapes
:??*
dtype02I
Gtrain/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/Conv2D/ReadVariableOp?
8train/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/Conv2DConv2DEtrain/MobilenetV1/MobilenetV1/Conv2d_10_depthwise/Relu6:activations:0Otrain/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
2:
8train/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/Conv2D?
Jtrain/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/BatchNorm/ReadVariableOpReadVariableOpStrain_mobilenetv1_mobilenetv1_conv2d_10_pointwise_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02L
Jtrain/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/BatchNorm/ReadVariableOp?
Ltrain/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/BatchNorm/ReadVariableOp_1ReadVariableOpUtrain_mobilenetv1_mobilenetv1_conv2d_10_pointwise_batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype02N
Ltrain/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/BatchNorm/ReadVariableOp_1?
[train/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOpReadVariableOpdtrain_mobilenetv1_mobilenetv1_conv2d_10_pointwise_batchnorm_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02]
[train/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp?
]train/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpftrain_mobilenetv1_mobilenetv1_conv2d_10_pointwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02_
]train/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1?
Ltrain/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/BatchNorm/FusedBatchNormV3FusedBatchNormV3Atrain/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/Conv2D:output:0Rtrain/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/BatchNorm/ReadVariableOp:value:0Ttrain/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/BatchNorm/ReadVariableOp_1:value:0ctrain/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp:value:0etrain/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:2N
Ltrain/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/BatchNorm/FusedBatchNormV3?
Qtrain/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/BatchNorm/AssignMovingAvg/sub_1Subbtrain/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/BatchNorm/AssignMovingAvg/ReadVariableOp:value:0Ytrain/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/BatchNorm/FusedBatchNormV3:batch_mean:0*
T0*
_output_shapes	
:?2S
Qtrain/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/BatchNorm/AssignMovingAvg/sub_1?
Qtrain/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/BatchNorm/AssignMovingAvg/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2S
Qtrain/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/BatchNorm/AssignMovingAvg/sub/x?
Otrain/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/BatchNorm/AssignMovingAvg/subSubZtrain/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/BatchNorm/AssignMovingAvg/sub/x:output:0placeholder_2*
T0*
_output_shapes
: 2Q
Otrain/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/BatchNorm/AssignMovingAvg/sub?
Otrain/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/BatchNorm/AssignMovingAvg/mulMulUtrain/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/BatchNorm/AssignMovingAvg/sub_1:z:0Strain/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/BatchNorm/AssignMovingAvg/sub:z:0*
T0*
_output_shapes	
:?2Q
Otrain/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/BatchNorm/AssignMovingAvg/mul?
Ktrain/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/BatchNorm/AssignMovingAvgAssignSubVariableOpdtrain_mobilenetv1_mobilenetv1_conv2d_10_pointwise_batchnorm_fusedbatchnormv3_readvariableop_resourceStrain/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/BatchNorm/AssignMovingAvg/mul:z:0*
_output_shapes
 *
dtype02M
Ktrain/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/BatchNorm/AssignMovingAvg?
\train/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/BatchNorm/AssignMovingAvg_1/ReadVariableOpReadVariableOpftrain_mobilenetv1_mobilenetv1_conv2d_10_pointwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02^
\train/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/BatchNorm/AssignMovingAvg_1/ReadVariableOp?
Strain/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/BatchNorm/AssignMovingAvg_1/sub_1Subdtrain/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/BatchNorm/AssignMovingAvg_1/ReadVariableOp:value:0]train/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/BatchNorm/FusedBatchNormV3:batch_variance:0*
T0*
_output_shapes	
:?2U
Strain/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/BatchNorm/AssignMovingAvg_1/sub_1?
Strain/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/BatchNorm/AssignMovingAvg_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2U
Strain/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/BatchNorm/AssignMovingAvg_1/sub/x?
Qtrain/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/BatchNorm/AssignMovingAvg_1/subSub\train/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/BatchNorm/AssignMovingAvg_1/sub/x:output:0placeholder_2*
T0*
_output_shapes
: 2S
Qtrain/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/BatchNorm/AssignMovingAvg_1/sub?
Qtrain/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/BatchNorm/AssignMovingAvg_1/mulMulWtrain/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/BatchNorm/AssignMovingAvg_1/sub_1:z:0Utrain/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/BatchNorm/AssignMovingAvg_1/sub:z:0*
T0*
_output_shapes	
:?2S
Qtrain/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/BatchNorm/AssignMovingAvg_1/mul?
Mtrain/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/BatchNorm/AssignMovingAvg_1AssignSubVariableOpftrain_mobilenetv1_mobilenetv1_conv2d_10_pointwise_batchnorm_fusedbatchnormv3_readvariableop_1_resourceUtrain/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/BatchNorm/AssignMovingAvg_1/mul:z:0*
_output_shapes
 *
dtype02O
Mtrain/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/BatchNorm/AssignMovingAvg_1?
Ztrain/MobilenetV1/MobilenetV1/Conv2d_11_depthwise/BatchNorm/AssignMovingAvg/ReadVariableOpReadVariableOpdtrain_mobilenetv1_mobilenetv1_conv2d_11_depthwise_batchnorm_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02\
Ztrain/MobilenetV1/MobilenetV1/Conv2d_11_depthwise/BatchNorm/AssignMovingAvg/ReadVariableOp?
7train/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/Relu6Relu6Ptrain/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/BatchNorm/FusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,????????????????????????????29
7train/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/Relu6?
Jtrain/MobilenetV1/MobilenetV1/Conv2d_11_depthwise/depthwise/ReadVariableOpReadVariableOpStrain_mobilenetv1_mobilenetv1_conv2d_11_depthwise_depthwise_readvariableop_resource*'
_output_shapes
:?*
dtype02L
Jtrain/MobilenetV1/MobilenetV1/Conv2d_11_depthwise/depthwise/ReadVariableOp?
;train/MobilenetV1/MobilenetV1/Conv2d_11_depthwise/depthwiseDepthwiseConv2dNativeEtrain/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/Relu6:activations:0Rtrain/MobilenetV1/MobilenetV1/Conv2d_11_depthwise/depthwise/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
2=
;train/MobilenetV1/MobilenetV1/Conv2d_11_depthwise/depthwise?
Jtrain/MobilenetV1/MobilenetV1/Conv2d_11_depthwise/BatchNorm/ReadVariableOpReadVariableOpStrain_mobilenetv1_mobilenetv1_conv2d_11_depthwise_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02L
Jtrain/MobilenetV1/MobilenetV1/Conv2d_11_depthwise/BatchNorm/ReadVariableOp?
Ltrain/MobilenetV1/MobilenetV1/Conv2d_11_depthwise/BatchNorm/ReadVariableOp_1ReadVariableOpUtrain_mobilenetv1_mobilenetv1_conv2d_11_depthwise_batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype02N
Ltrain/MobilenetV1/MobilenetV1/Conv2d_11_depthwise/BatchNorm/ReadVariableOp_1?
[train/MobilenetV1/MobilenetV1/Conv2d_11_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOpReadVariableOpdtrain_mobilenetv1_mobilenetv1_conv2d_11_depthwise_batchnorm_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02]
[train/MobilenetV1/MobilenetV1/Conv2d_11_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp?
]train/MobilenetV1/MobilenetV1/Conv2d_11_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpftrain_mobilenetv1_mobilenetv1_conv2d_11_depthwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02_
]train/MobilenetV1/MobilenetV1/Conv2d_11_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1?
Ltrain/MobilenetV1/MobilenetV1/Conv2d_11_depthwise/BatchNorm/FusedBatchNormV3FusedBatchNormV3Dtrain/MobilenetV1/MobilenetV1/Conv2d_11_depthwise/depthwise:output:0Rtrain/MobilenetV1/MobilenetV1/Conv2d_11_depthwise/BatchNorm/ReadVariableOp:value:0Ttrain/MobilenetV1/MobilenetV1/Conv2d_11_depthwise/BatchNorm/ReadVariableOp_1:value:0ctrain/MobilenetV1/MobilenetV1/Conv2d_11_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp:value:0etrain/MobilenetV1/MobilenetV1/Conv2d_11_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:2N
Ltrain/MobilenetV1/MobilenetV1/Conv2d_11_depthwise/BatchNorm/FusedBatchNormV3?
Qtrain/MobilenetV1/MobilenetV1/Conv2d_11_depthwise/BatchNorm/AssignMovingAvg/sub_1Subbtrain/MobilenetV1/MobilenetV1/Conv2d_11_depthwise/BatchNorm/AssignMovingAvg/ReadVariableOp:value:0Ytrain/MobilenetV1/MobilenetV1/Conv2d_11_depthwise/BatchNorm/FusedBatchNormV3:batch_mean:0*
T0*
_output_shapes	
:?2S
Qtrain/MobilenetV1/MobilenetV1/Conv2d_11_depthwise/BatchNorm/AssignMovingAvg/sub_1?
Qtrain/MobilenetV1/MobilenetV1/Conv2d_11_depthwise/BatchNorm/AssignMovingAvg/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2S
Qtrain/MobilenetV1/MobilenetV1/Conv2d_11_depthwise/BatchNorm/AssignMovingAvg/sub/x?
Otrain/MobilenetV1/MobilenetV1/Conv2d_11_depthwise/BatchNorm/AssignMovingAvg/subSubZtrain/MobilenetV1/MobilenetV1/Conv2d_11_depthwise/BatchNorm/AssignMovingAvg/sub/x:output:0placeholder_2*
T0*
_output_shapes
: 2Q
Otrain/MobilenetV1/MobilenetV1/Conv2d_11_depthwise/BatchNorm/AssignMovingAvg/sub?
Otrain/MobilenetV1/MobilenetV1/Conv2d_11_depthwise/BatchNorm/AssignMovingAvg/mulMulUtrain/MobilenetV1/MobilenetV1/Conv2d_11_depthwise/BatchNorm/AssignMovingAvg/sub_1:z:0Strain/MobilenetV1/MobilenetV1/Conv2d_11_depthwise/BatchNorm/AssignMovingAvg/sub:z:0*
T0*
_output_shapes	
:?2Q
Otrain/MobilenetV1/MobilenetV1/Conv2d_11_depthwise/BatchNorm/AssignMovingAvg/mul?
Ktrain/MobilenetV1/MobilenetV1/Conv2d_11_depthwise/BatchNorm/AssignMovingAvgAssignSubVariableOpdtrain_mobilenetv1_mobilenetv1_conv2d_11_depthwise_batchnorm_fusedbatchnormv3_readvariableop_resourceStrain/MobilenetV1/MobilenetV1/Conv2d_11_depthwise/BatchNorm/AssignMovingAvg/mul:z:0*
_output_shapes
 *
dtype02M
Ktrain/MobilenetV1/MobilenetV1/Conv2d_11_depthwise/BatchNorm/AssignMovingAvg?
\train/MobilenetV1/MobilenetV1/Conv2d_11_depthwise/BatchNorm/AssignMovingAvg_1/ReadVariableOpReadVariableOpftrain_mobilenetv1_mobilenetv1_conv2d_11_depthwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02^
\train/MobilenetV1/MobilenetV1/Conv2d_11_depthwise/BatchNorm/AssignMovingAvg_1/ReadVariableOp?
Strain/MobilenetV1/MobilenetV1/Conv2d_11_depthwise/BatchNorm/AssignMovingAvg_1/sub_1Subdtrain/MobilenetV1/MobilenetV1/Conv2d_11_depthwise/BatchNorm/AssignMovingAvg_1/ReadVariableOp:value:0]train/MobilenetV1/MobilenetV1/Conv2d_11_depthwise/BatchNorm/FusedBatchNormV3:batch_variance:0*
T0*
_output_shapes	
:?2U
Strain/MobilenetV1/MobilenetV1/Conv2d_11_depthwise/BatchNorm/AssignMovingAvg_1/sub_1?
Strain/MobilenetV1/MobilenetV1/Conv2d_11_depthwise/BatchNorm/AssignMovingAvg_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2U
Strain/MobilenetV1/MobilenetV1/Conv2d_11_depthwise/BatchNorm/AssignMovingAvg_1/sub/x?
Qtrain/MobilenetV1/MobilenetV1/Conv2d_11_depthwise/BatchNorm/AssignMovingAvg_1/subSub\train/MobilenetV1/MobilenetV1/Conv2d_11_depthwise/BatchNorm/AssignMovingAvg_1/sub/x:output:0placeholder_2*
T0*
_output_shapes
: 2S
Qtrain/MobilenetV1/MobilenetV1/Conv2d_11_depthwise/BatchNorm/AssignMovingAvg_1/sub?
Qtrain/MobilenetV1/MobilenetV1/Conv2d_11_depthwise/BatchNorm/AssignMovingAvg_1/mulMulWtrain/MobilenetV1/MobilenetV1/Conv2d_11_depthwise/BatchNorm/AssignMovingAvg_1/sub_1:z:0Utrain/MobilenetV1/MobilenetV1/Conv2d_11_depthwise/BatchNorm/AssignMovingAvg_1/sub:z:0*
T0*
_output_shapes	
:?2S
Qtrain/MobilenetV1/MobilenetV1/Conv2d_11_depthwise/BatchNorm/AssignMovingAvg_1/mul?
Mtrain/MobilenetV1/MobilenetV1/Conv2d_11_depthwise/BatchNorm/AssignMovingAvg_1AssignSubVariableOpftrain_mobilenetv1_mobilenetv1_conv2d_11_depthwise_batchnorm_fusedbatchnormv3_readvariableop_1_resourceUtrain/MobilenetV1/MobilenetV1/Conv2d_11_depthwise/BatchNorm/AssignMovingAvg_1/mul:z:0*
_output_shapes
 *
dtype02O
Mtrain/MobilenetV1/MobilenetV1/Conv2d_11_depthwise/BatchNorm/AssignMovingAvg_1?
Ztrain/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/BatchNorm/AssignMovingAvg/ReadVariableOpReadVariableOpdtrain_mobilenetv1_mobilenetv1_conv2d_11_pointwise_batchnorm_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02\
Ztrain/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/BatchNorm/AssignMovingAvg/ReadVariableOp?
7train/MobilenetV1/MobilenetV1/Conv2d_11_depthwise/Relu6Relu6Ptrain/MobilenetV1/MobilenetV1/Conv2d_11_depthwise/BatchNorm/FusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,????????????????????????????29
7train/MobilenetV1/MobilenetV1/Conv2d_11_depthwise/Relu6?
Gtrain/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/Conv2D/ReadVariableOpReadVariableOprtrain_mobilenetv1_mobilenetv1_conv2d_11_pointwise_kernel_regularizer_l2_regularizer_l2loss_readvariableop_resource*(
_output_shapes
:??*
dtype02I
Gtrain/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/Conv2D/ReadVariableOp?
8train/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/Conv2DConv2DEtrain/MobilenetV1/MobilenetV1/Conv2d_11_depthwise/Relu6:activations:0Otrain/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
2:
8train/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/Conv2D?
Jtrain/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/BatchNorm/ReadVariableOpReadVariableOpStrain_mobilenetv1_mobilenetv1_conv2d_11_pointwise_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02L
Jtrain/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/BatchNorm/ReadVariableOp?
Ltrain/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/BatchNorm/ReadVariableOp_1ReadVariableOpUtrain_mobilenetv1_mobilenetv1_conv2d_11_pointwise_batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype02N
Ltrain/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/BatchNorm/ReadVariableOp_1?
[train/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOpReadVariableOpdtrain_mobilenetv1_mobilenetv1_conv2d_11_pointwise_batchnorm_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02]
[train/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp?
]train/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpftrain_mobilenetv1_mobilenetv1_conv2d_11_pointwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02_
]train/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1?
Ltrain/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/BatchNorm/FusedBatchNormV3FusedBatchNormV3Atrain/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/Conv2D:output:0Rtrain/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/BatchNorm/ReadVariableOp:value:0Ttrain/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/BatchNorm/ReadVariableOp_1:value:0ctrain/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp:value:0etrain/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:2N
Ltrain/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/BatchNorm/FusedBatchNormV3?
Qtrain/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/BatchNorm/AssignMovingAvg/sub_1Subbtrain/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/BatchNorm/AssignMovingAvg/ReadVariableOp:value:0Ytrain/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/BatchNorm/FusedBatchNormV3:batch_mean:0*
T0*
_output_shapes	
:?2S
Qtrain/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/BatchNorm/AssignMovingAvg/sub_1?
Qtrain/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/BatchNorm/AssignMovingAvg/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2S
Qtrain/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/BatchNorm/AssignMovingAvg/sub/x?
Otrain/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/BatchNorm/AssignMovingAvg/subSubZtrain/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/BatchNorm/AssignMovingAvg/sub/x:output:0placeholder_2*
T0*
_output_shapes
: 2Q
Otrain/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/BatchNorm/AssignMovingAvg/sub?
Otrain/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/BatchNorm/AssignMovingAvg/mulMulUtrain/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/BatchNorm/AssignMovingAvg/sub_1:z:0Strain/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/BatchNorm/AssignMovingAvg/sub:z:0*
T0*
_output_shapes	
:?2Q
Otrain/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/BatchNorm/AssignMovingAvg/mul?
Ktrain/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/BatchNorm/AssignMovingAvgAssignSubVariableOpdtrain_mobilenetv1_mobilenetv1_conv2d_11_pointwise_batchnorm_fusedbatchnormv3_readvariableop_resourceStrain/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/BatchNorm/AssignMovingAvg/mul:z:0*
_output_shapes
 *
dtype02M
Ktrain/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/BatchNorm/AssignMovingAvg?
\train/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/BatchNorm/AssignMovingAvg_1/ReadVariableOpReadVariableOpftrain_mobilenetv1_mobilenetv1_conv2d_11_pointwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02^
\train/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/BatchNorm/AssignMovingAvg_1/ReadVariableOp?
Strain/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/BatchNorm/AssignMovingAvg_1/sub_1Subdtrain/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/BatchNorm/AssignMovingAvg_1/ReadVariableOp:value:0]train/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/BatchNorm/FusedBatchNormV3:batch_variance:0*
T0*
_output_shapes	
:?2U
Strain/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/BatchNorm/AssignMovingAvg_1/sub_1?
Strain/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/BatchNorm/AssignMovingAvg_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2U
Strain/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/BatchNorm/AssignMovingAvg_1/sub/x?
Qtrain/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/BatchNorm/AssignMovingAvg_1/subSub\train/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/BatchNorm/AssignMovingAvg_1/sub/x:output:0placeholder_2*
T0*
_output_shapes
: 2S
Qtrain/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/BatchNorm/AssignMovingAvg_1/sub?
Qtrain/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/BatchNorm/AssignMovingAvg_1/mulMulWtrain/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/BatchNorm/AssignMovingAvg_1/sub_1:z:0Utrain/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/BatchNorm/AssignMovingAvg_1/sub:z:0*
T0*
_output_shapes	
:?2S
Qtrain/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/BatchNorm/AssignMovingAvg_1/mul?
Mtrain/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/BatchNorm/AssignMovingAvg_1AssignSubVariableOpftrain_mobilenetv1_mobilenetv1_conv2d_11_pointwise_batchnorm_fusedbatchnormv3_readvariableop_1_resourceUtrain/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/BatchNorm/AssignMovingAvg_1/mul:z:0*
_output_shapes
 *
dtype02O
Mtrain/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/BatchNorm/AssignMovingAvg_1?
Ztrain/MobilenetV1/MobilenetV1/Conv2d_12_depthwise/BatchNorm/AssignMovingAvg/ReadVariableOpReadVariableOpdtrain_mobilenetv1_mobilenetv1_conv2d_12_depthwise_batchnorm_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02\
Ztrain/MobilenetV1/MobilenetV1/Conv2d_12_depthwise/BatchNorm/AssignMovingAvg/ReadVariableOp?
7train/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/Relu6Relu6Ptrain/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/BatchNorm/FusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,????????????????????????????29
7train/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/Relu6?
Jtrain/MobilenetV1/MobilenetV1/Conv2d_12_depthwise/depthwise/ReadVariableOpReadVariableOpStrain_mobilenetv1_mobilenetv1_conv2d_12_depthwise_depthwise_readvariableop_resource*'
_output_shapes
:?*
dtype02L
Jtrain/MobilenetV1/MobilenetV1/Conv2d_12_depthwise/depthwise/ReadVariableOp?
;train/MobilenetV1/MobilenetV1/Conv2d_12_depthwise/depthwiseDepthwiseConv2dNativeEtrain/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/Relu6:activations:0Rtrain/MobilenetV1/MobilenetV1/Conv2d_12_depthwise/depthwise/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
2=
;train/MobilenetV1/MobilenetV1/Conv2d_12_depthwise/depthwise?
Jtrain/MobilenetV1/MobilenetV1/Conv2d_12_depthwise/BatchNorm/ReadVariableOpReadVariableOpStrain_mobilenetv1_mobilenetv1_conv2d_12_depthwise_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02L
Jtrain/MobilenetV1/MobilenetV1/Conv2d_12_depthwise/BatchNorm/ReadVariableOp?
Ltrain/MobilenetV1/MobilenetV1/Conv2d_12_depthwise/BatchNorm/ReadVariableOp_1ReadVariableOpUtrain_mobilenetv1_mobilenetv1_conv2d_12_depthwise_batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype02N
Ltrain/MobilenetV1/MobilenetV1/Conv2d_12_depthwise/BatchNorm/ReadVariableOp_1?
[train/MobilenetV1/MobilenetV1/Conv2d_12_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOpReadVariableOpdtrain_mobilenetv1_mobilenetv1_conv2d_12_depthwise_batchnorm_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02]
[train/MobilenetV1/MobilenetV1/Conv2d_12_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp?
]train/MobilenetV1/MobilenetV1/Conv2d_12_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpftrain_mobilenetv1_mobilenetv1_conv2d_12_depthwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02_
]train/MobilenetV1/MobilenetV1/Conv2d_12_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1?
Ltrain/MobilenetV1/MobilenetV1/Conv2d_12_depthwise/BatchNorm/FusedBatchNormV3FusedBatchNormV3Dtrain/MobilenetV1/MobilenetV1/Conv2d_12_depthwise/depthwise:output:0Rtrain/MobilenetV1/MobilenetV1/Conv2d_12_depthwise/BatchNorm/ReadVariableOp:value:0Ttrain/MobilenetV1/MobilenetV1/Conv2d_12_depthwise/BatchNorm/ReadVariableOp_1:value:0ctrain/MobilenetV1/MobilenetV1/Conv2d_12_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp:value:0etrain/MobilenetV1/MobilenetV1/Conv2d_12_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:2N
Ltrain/MobilenetV1/MobilenetV1/Conv2d_12_depthwise/BatchNorm/FusedBatchNormV3?
Qtrain/MobilenetV1/MobilenetV1/Conv2d_12_depthwise/BatchNorm/AssignMovingAvg/sub_1Subbtrain/MobilenetV1/MobilenetV1/Conv2d_12_depthwise/BatchNorm/AssignMovingAvg/ReadVariableOp:value:0Ytrain/MobilenetV1/MobilenetV1/Conv2d_12_depthwise/BatchNorm/FusedBatchNormV3:batch_mean:0*
T0*
_output_shapes	
:?2S
Qtrain/MobilenetV1/MobilenetV1/Conv2d_12_depthwise/BatchNorm/AssignMovingAvg/sub_1?
Qtrain/MobilenetV1/MobilenetV1/Conv2d_12_depthwise/BatchNorm/AssignMovingAvg/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2S
Qtrain/MobilenetV1/MobilenetV1/Conv2d_12_depthwise/BatchNorm/AssignMovingAvg/sub/x?
Otrain/MobilenetV1/MobilenetV1/Conv2d_12_depthwise/BatchNorm/AssignMovingAvg/subSubZtrain/MobilenetV1/MobilenetV1/Conv2d_12_depthwise/BatchNorm/AssignMovingAvg/sub/x:output:0placeholder_2*
T0*
_output_shapes
: 2Q
Otrain/MobilenetV1/MobilenetV1/Conv2d_12_depthwise/BatchNorm/AssignMovingAvg/sub?
Otrain/MobilenetV1/MobilenetV1/Conv2d_12_depthwise/BatchNorm/AssignMovingAvg/mulMulUtrain/MobilenetV1/MobilenetV1/Conv2d_12_depthwise/BatchNorm/AssignMovingAvg/sub_1:z:0Strain/MobilenetV1/MobilenetV1/Conv2d_12_depthwise/BatchNorm/AssignMovingAvg/sub:z:0*
T0*
_output_shapes	
:?2Q
Otrain/MobilenetV1/MobilenetV1/Conv2d_12_depthwise/BatchNorm/AssignMovingAvg/mul?
Ktrain/MobilenetV1/MobilenetV1/Conv2d_12_depthwise/BatchNorm/AssignMovingAvgAssignSubVariableOpdtrain_mobilenetv1_mobilenetv1_conv2d_12_depthwise_batchnorm_fusedbatchnormv3_readvariableop_resourceStrain/MobilenetV1/MobilenetV1/Conv2d_12_depthwise/BatchNorm/AssignMovingAvg/mul:z:0*
_output_shapes
 *
dtype02M
Ktrain/MobilenetV1/MobilenetV1/Conv2d_12_depthwise/BatchNorm/AssignMovingAvg?
\train/MobilenetV1/MobilenetV1/Conv2d_12_depthwise/BatchNorm/AssignMovingAvg_1/ReadVariableOpReadVariableOpftrain_mobilenetv1_mobilenetv1_conv2d_12_depthwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02^
\train/MobilenetV1/MobilenetV1/Conv2d_12_depthwise/BatchNorm/AssignMovingAvg_1/ReadVariableOp?
Strain/MobilenetV1/MobilenetV1/Conv2d_12_depthwise/BatchNorm/AssignMovingAvg_1/sub_1Subdtrain/MobilenetV1/MobilenetV1/Conv2d_12_depthwise/BatchNorm/AssignMovingAvg_1/ReadVariableOp:value:0]train/MobilenetV1/MobilenetV1/Conv2d_12_depthwise/BatchNorm/FusedBatchNormV3:batch_variance:0*
T0*
_output_shapes	
:?2U
Strain/MobilenetV1/MobilenetV1/Conv2d_12_depthwise/BatchNorm/AssignMovingAvg_1/sub_1?
Strain/MobilenetV1/MobilenetV1/Conv2d_12_depthwise/BatchNorm/AssignMovingAvg_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2U
Strain/MobilenetV1/MobilenetV1/Conv2d_12_depthwise/BatchNorm/AssignMovingAvg_1/sub/x?
Qtrain/MobilenetV1/MobilenetV1/Conv2d_12_depthwise/BatchNorm/AssignMovingAvg_1/subSub\train/MobilenetV1/MobilenetV1/Conv2d_12_depthwise/BatchNorm/AssignMovingAvg_1/sub/x:output:0placeholder_2*
T0*
_output_shapes
: 2S
Qtrain/MobilenetV1/MobilenetV1/Conv2d_12_depthwise/BatchNorm/AssignMovingAvg_1/sub?
Qtrain/MobilenetV1/MobilenetV1/Conv2d_12_depthwise/BatchNorm/AssignMovingAvg_1/mulMulWtrain/MobilenetV1/MobilenetV1/Conv2d_12_depthwise/BatchNorm/AssignMovingAvg_1/sub_1:z:0Utrain/MobilenetV1/MobilenetV1/Conv2d_12_depthwise/BatchNorm/AssignMovingAvg_1/sub:z:0*
T0*
_output_shapes	
:?2S
Qtrain/MobilenetV1/MobilenetV1/Conv2d_12_depthwise/BatchNorm/AssignMovingAvg_1/mul?
Mtrain/MobilenetV1/MobilenetV1/Conv2d_12_depthwise/BatchNorm/AssignMovingAvg_1AssignSubVariableOpftrain_mobilenetv1_mobilenetv1_conv2d_12_depthwise_batchnorm_fusedbatchnormv3_readvariableop_1_resourceUtrain/MobilenetV1/MobilenetV1/Conv2d_12_depthwise/BatchNorm/AssignMovingAvg_1/mul:z:0*
_output_shapes
 *
dtype02O
Mtrain/MobilenetV1/MobilenetV1/Conv2d_12_depthwise/BatchNorm/AssignMovingAvg_1?
Ztrain/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/BatchNorm/AssignMovingAvg/ReadVariableOpReadVariableOpdtrain_mobilenetv1_mobilenetv1_conv2d_12_pointwise_batchnorm_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02\
Ztrain/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/BatchNorm/AssignMovingAvg/ReadVariableOp?
7train/MobilenetV1/MobilenetV1/Conv2d_12_depthwise/Relu6Relu6Ptrain/MobilenetV1/MobilenetV1/Conv2d_12_depthwise/BatchNorm/FusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,????????????????????????????29
7train/MobilenetV1/MobilenetV1/Conv2d_12_depthwise/Relu6?
Gtrain/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/Conv2D/ReadVariableOpReadVariableOprtrain_mobilenetv1_mobilenetv1_conv2d_12_pointwise_kernel_regularizer_l2_regularizer_l2loss_readvariableop_resource*(
_output_shapes
:??*
dtype02I
Gtrain/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/Conv2D/ReadVariableOp?
8train/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/Conv2DConv2DEtrain/MobilenetV1/MobilenetV1/Conv2d_12_depthwise/Relu6:activations:0Otrain/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
2:
8train/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/Conv2D?
Jtrain/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/BatchNorm/ReadVariableOpReadVariableOpStrain_mobilenetv1_mobilenetv1_conv2d_12_pointwise_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02L
Jtrain/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/BatchNorm/ReadVariableOp?
Ltrain/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/BatchNorm/ReadVariableOp_1ReadVariableOpUtrain_mobilenetv1_mobilenetv1_conv2d_12_pointwise_batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype02N
Ltrain/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/BatchNorm/ReadVariableOp_1?
[train/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOpReadVariableOpdtrain_mobilenetv1_mobilenetv1_conv2d_12_pointwise_batchnorm_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02]
[train/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp?
]train/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpftrain_mobilenetv1_mobilenetv1_conv2d_12_pointwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02_
]train/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1?
Ltrain/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/BatchNorm/FusedBatchNormV3FusedBatchNormV3Atrain/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/Conv2D:output:0Rtrain/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/BatchNorm/ReadVariableOp:value:0Ttrain/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/BatchNorm/ReadVariableOp_1:value:0ctrain/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp:value:0etrain/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:2N
Ltrain/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/BatchNorm/FusedBatchNormV3?
Qtrain/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/BatchNorm/AssignMovingAvg/sub_1Subbtrain/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/BatchNorm/AssignMovingAvg/ReadVariableOp:value:0Ytrain/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/BatchNorm/FusedBatchNormV3:batch_mean:0*
T0*
_output_shapes	
:?2S
Qtrain/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/BatchNorm/AssignMovingAvg/sub_1?
Qtrain/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/BatchNorm/AssignMovingAvg/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2S
Qtrain/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/BatchNorm/AssignMovingAvg/sub/x?
Otrain/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/BatchNorm/AssignMovingAvg/subSubZtrain/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/BatchNorm/AssignMovingAvg/sub/x:output:0placeholder_2*
T0*
_output_shapes
: 2Q
Otrain/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/BatchNorm/AssignMovingAvg/sub?
Otrain/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/BatchNorm/AssignMovingAvg/mulMulUtrain/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/BatchNorm/AssignMovingAvg/sub_1:z:0Strain/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/BatchNorm/AssignMovingAvg/sub:z:0*
T0*
_output_shapes	
:?2Q
Otrain/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/BatchNorm/AssignMovingAvg/mul?
Ktrain/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/BatchNorm/AssignMovingAvgAssignSubVariableOpdtrain_mobilenetv1_mobilenetv1_conv2d_12_pointwise_batchnorm_fusedbatchnormv3_readvariableop_resourceStrain/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/BatchNorm/AssignMovingAvg/mul:z:0*
_output_shapes
 *
dtype02M
Ktrain/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/BatchNorm/AssignMovingAvg?
\train/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/BatchNorm/AssignMovingAvg_1/ReadVariableOpReadVariableOpftrain_mobilenetv1_mobilenetv1_conv2d_12_pointwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02^
\train/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/BatchNorm/AssignMovingAvg_1/ReadVariableOp?
Strain/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/BatchNorm/AssignMovingAvg_1/sub_1Subdtrain/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/BatchNorm/AssignMovingAvg_1/ReadVariableOp:value:0]train/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/BatchNorm/FusedBatchNormV3:batch_variance:0*
T0*
_output_shapes	
:?2U
Strain/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/BatchNorm/AssignMovingAvg_1/sub_1?
Strain/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/BatchNorm/AssignMovingAvg_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2U
Strain/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/BatchNorm/AssignMovingAvg_1/sub/x?
Qtrain/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/BatchNorm/AssignMovingAvg_1/subSub\train/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/BatchNorm/AssignMovingAvg_1/sub/x:output:0placeholder_2*
T0*
_output_shapes
: 2S
Qtrain/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/BatchNorm/AssignMovingAvg_1/sub?
Qtrain/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/BatchNorm/AssignMovingAvg_1/mulMulWtrain/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/BatchNorm/AssignMovingAvg_1/sub_1:z:0Utrain/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/BatchNorm/AssignMovingAvg_1/sub:z:0*
T0*
_output_shapes	
:?2S
Qtrain/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/BatchNorm/AssignMovingAvg_1/mul?
Mtrain/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/BatchNorm/AssignMovingAvg_1AssignSubVariableOpftrain_mobilenetv1_mobilenetv1_conv2d_12_pointwise_batchnorm_fusedbatchnormv3_readvariableop_1_resourceUtrain/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/BatchNorm/AssignMovingAvg_1/mul:z:0*
_output_shapes
 *
dtype02O
Mtrain/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/BatchNorm/AssignMovingAvg_1?
Ztrain/MobilenetV1/MobilenetV1/Conv2d_13_depthwise/BatchNorm/AssignMovingAvg/ReadVariableOpReadVariableOpdtrain_mobilenetv1_mobilenetv1_conv2d_13_depthwise_batchnorm_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02\
Ztrain/MobilenetV1/MobilenetV1/Conv2d_13_depthwise/BatchNorm/AssignMovingAvg/ReadVariableOp?
7train/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/Relu6Relu6Ptrain/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/BatchNorm/FusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,????????????????????????????29
7train/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/Relu6?
Jtrain/MobilenetV1/MobilenetV1/Conv2d_13_depthwise/depthwise/ReadVariableOpReadVariableOpStrain_mobilenetv1_mobilenetv1_conv2d_13_depthwise_depthwise_readvariableop_resource*'
_output_shapes
:?*
dtype02L
Jtrain/MobilenetV1/MobilenetV1/Conv2d_13_depthwise/depthwise/ReadVariableOp?
;train/MobilenetV1/MobilenetV1/Conv2d_13_depthwise/depthwiseDepthwiseConv2dNativeEtrain/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/Relu6:activations:0Rtrain/MobilenetV1/MobilenetV1/Conv2d_13_depthwise/depthwise/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
2=
;train/MobilenetV1/MobilenetV1/Conv2d_13_depthwise/depthwise?
Jtrain/MobilenetV1/MobilenetV1/Conv2d_13_depthwise/BatchNorm/ReadVariableOpReadVariableOpStrain_mobilenetv1_mobilenetv1_conv2d_13_depthwise_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02L
Jtrain/MobilenetV1/MobilenetV1/Conv2d_13_depthwise/BatchNorm/ReadVariableOp?
Ltrain/MobilenetV1/MobilenetV1/Conv2d_13_depthwise/BatchNorm/ReadVariableOp_1ReadVariableOpUtrain_mobilenetv1_mobilenetv1_conv2d_13_depthwise_batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype02N
Ltrain/MobilenetV1/MobilenetV1/Conv2d_13_depthwise/BatchNorm/ReadVariableOp_1?
[train/MobilenetV1/MobilenetV1/Conv2d_13_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOpReadVariableOpdtrain_mobilenetv1_mobilenetv1_conv2d_13_depthwise_batchnorm_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02]
[train/MobilenetV1/MobilenetV1/Conv2d_13_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp?
]train/MobilenetV1/MobilenetV1/Conv2d_13_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpftrain_mobilenetv1_mobilenetv1_conv2d_13_depthwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02_
]train/MobilenetV1/MobilenetV1/Conv2d_13_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1?
Ltrain/MobilenetV1/MobilenetV1/Conv2d_13_depthwise/BatchNorm/FusedBatchNormV3FusedBatchNormV3Dtrain/MobilenetV1/MobilenetV1/Conv2d_13_depthwise/depthwise:output:0Rtrain/MobilenetV1/MobilenetV1/Conv2d_13_depthwise/BatchNorm/ReadVariableOp:value:0Ttrain/MobilenetV1/MobilenetV1/Conv2d_13_depthwise/BatchNorm/ReadVariableOp_1:value:0ctrain/MobilenetV1/MobilenetV1/Conv2d_13_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp:value:0etrain/MobilenetV1/MobilenetV1/Conv2d_13_depthwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:2N
Ltrain/MobilenetV1/MobilenetV1/Conv2d_13_depthwise/BatchNorm/FusedBatchNormV3?
Qtrain/MobilenetV1/MobilenetV1/Conv2d_13_depthwise/BatchNorm/AssignMovingAvg/sub_1Subbtrain/MobilenetV1/MobilenetV1/Conv2d_13_depthwise/BatchNorm/AssignMovingAvg/ReadVariableOp:value:0Ytrain/MobilenetV1/MobilenetV1/Conv2d_13_depthwise/BatchNorm/FusedBatchNormV3:batch_mean:0*
T0*
_output_shapes	
:?2S
Qtrain/MobilenetV1/MobilenetV1/Conv2d_13_depthwise/BatchNorm/AssignMovingAvg/sub_1?
Qtrain/MobilenetV1/MobilenetV1/Conv2d_13_depthwise/BatchNorm/AssignMovingAvg/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2S
Qtrain/MobilenetV1/MobilenetV1/Conv2d_13_depthwise/BatchNorm/AssignMovingAvg/sub/x?
Otrain/MobilenetV1/MobilenetV1/Conv2d_13_depthwise/BatchNorm/AssignMovingAvg/subSubZtrain/MobilenetV1/MobilenetV1/Conv2d_13_depthwise/BatchNorm/AssignMovingAvg/sub/x:output:0placeholder_2*
T0*
_output_shapes
: 2Q
Otrain/MobilenetV1/MobilenetV1/Conv2d_13_depthwise/BatchNorm/AssignMovingAvg/sub?
Otrain/MobilenetV1/MobilenetV1/Conv2d_13_depthwise/BatchNorm/AssignMovingAvg/mulMulUtrain/MobilenetV1/MobilenetV1/Conv2d_13_depthwise/BatchNorm/AssignMovingAvg/sub_1:z:0Strain/MobilenetV1/MobilenetV1/Conv2d_13_depthwise/BatchNorm/AssignMovingAvg/sub:z:0*
T0*
_output_shapes	
:?2Q
Otrain/MobilenetV1/MobilenetV1/Conv2d_13_depthwise/BatchNorm/AssignMovingAvg/mul?
Ktrain/MobilenetV1/MobilenetV1/Conv2d_13_depthwise/BatchNorm/AssignMovingAvgAssignSubVariableOpdtrain_mobilenetv1_mobilenetv1_conv2d_13_depthwise_batchnorm_fusedbatchnormv3_readvariableop_resourceStrain/MobilenetV1/MobilenetV1/Conv2d_13_depthwise/BatchNorm/AssignMovingAvg/mul:z:0*
_output_shapes
 *
dtype02M
Ktrain/MobilenetV1/MobilenetV1/Conv2d_13_depthwise/BatchNorm/AssignMovingAvg?
\train/MobilenetV1/MobilenetV1/Conv2d_13_depthwise/BatchNorm/AssignMovingAvg_1/ReadVariableOpReadVariableOpftrain_mobilenetv1_mobilenetv1_conv2d_13_depthwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02^
\train/MobilenetV1/MobilenetV1/Conv2d_13_depthwise/BatchNorm/AssignMovingAvg_1/ReadVariableOp?
Strain/MobilenetV1/MobilenetV1/Conv2d_13_depthwise/BatchNorm/AssignMovingAvg_1/sub_1Subdtrain/MobilenetV1/MobilenetV1/Conv2d_13_depthwise/BatchNorm/AssignMovingAvg_1/ReadVariableOp:value:0]train/MobilenetV1/MobilenetV1/Conv2d_13_depthwise/BatchNorm/FusedBatchNormV3:batch_variance:0*
T0*
_output_shapes	
:?2U
Strain/MobilenetV1/MobilenetV1/Conv2d_13_depthwise/BatchNorm/AssignMovingAvg_1/sub_1?
Strain/MobilenetV1/MobilenetV1/Conv2d_13_depthwise/BatchNorm/AssignMovingAvg_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2U
Strain/MobilenetV1/MobilenetV1/Conv2d_13_depthwise/BatchNorm/AssignMovingAvg_1/sub/x?
Qtrain/MobilenetV1/MobilenetV1/Conv2d_13_depthwise/BatchNorm/AssignMovingAvg_1/subSub\train/MobilenetV1/MobilenetV1/Conv2d_13_depthwise/BatchNorm/AssignMovingAvg_1/sub/x:output:0placeholder_2*
T0*
_output_shapes
: 2S
Qtrain/MobilenetV1/MobilenetV1/Conv2d_13_depthwise/BatchNorm/AssignMovingAvg_1/sub?
Qtrain/MobilenetV1/MobilenetV1/Conv2d_13_depthwise/BatchNorm/AssignMovingAvg_1/mulMulWtrain/MobilenetV1/MobilenetV1/Conv2d_13_depthwise/BatchNorm/AssignMovingAvg_1/sub_1:z:0Utrain/MobilenetV1/MobilenetV1/Conv2d_13_depthwise/BatchNorm/AssignMovingAvg_1/sub:z:0*
T0*
_output_shapes	
:?2S
Qtrain/MobilenetV1/MobilenetV1/Conv2d_13_depthwise/BatchNorm/AssignMovingAvg_1/mul?
Mtrain/MobilenetV1/MobilenetV1/Conv2d_13_depthwise/BatchNorm/AssignMovingAvg_1AssignSubVariableOpftrain_mobilenetv1_mobilenetv1_conv2d_13_depthwise_batchnorm_fusedbatchnormv3_readvariableop_1_resourceUtrain/MobilenetV1/MobilenetV1/Conv2d_13_depthwise/BatchNorm/AssignMovingAvg_1/mul:z:0*
_output_shapes
 *
dtype02O
Mtrain/MobilenetV1/MobilenetV1/Conv2d_13_depthwise/BatchNorm/AssignMovingAvg_1?
Ztrain/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/BatchNorm/AssignMovingAvg/ReadVariableOpReadVariableOpdtrain_mobilenetv1_mobilenetv1_conv2d_13_pointwise_batchnorm_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02\
Ztrain/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/BatchNorm/AssignMovingAvg/ReadVariableOp?
7train/MobilenetV1/MobilenetV1/Conv2d_13_depthwise/Relu6Relu6Ptrain/MobilenetV1/MobilenetV1/Conv2d_13_depthwise/BatchNorm/FusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,????????????????????????????29
7train/MobilenetV1/MobilenetV1/Conv2d_13_depthwise/Relu6?
Gtrain/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/Conv2D/ReadVariableOpReadVariableOprtrain_mobilenetv1_mobilenetv1_conv2d_13_pointwise_kernel_regularizer_l2_regularizer_l2loss_readvariableop_resource*(
_output_shapes
:??*
dtype02I
Gtrain/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/Conv2D/ReadVariableOp?
8train/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/Conv2DConv2DEtrain/MobilenetV1/MobilenetV1/Conv2d_13_depthwise/Relu6:activations:0Otrain/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
2:
8train/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/Conv2D?
Jtrain/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/BatchNorm/ReadVariableOpReadVariableOpStrain_mobilenetv1_mobilenetv1_conv2d_13_pointwise_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02L
Jtrain/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/BatchNorm/ReadVariableOp?
Ltrain/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/BatchNorm/ReadVariableOp_1ReadVariableOpUtrain_mobilenetv1_mobilenetv1_conv2d_13_pointwise_batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype02N
Ltrain/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/BatchNorm/ReadVariableOp_1?
[train/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOpReadVariableOpdtrain_mobilenetv1_mobilenetv1_conv2d_13_pointwise_batchnorm_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02]
[train/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp?
]train/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpftrain_mobilenetv1_mobilenetv1_conv2d_13_pointwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02_
]train/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1?
Ltrain/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/BatchNorm/FusedBatchNormV3FusedBatchNormV3Atrain/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/Conv2D:output:0Rtrain/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/BatchNorm/ReadVariableOp:value:0Ttrain/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/BatchNorm/ReadVariableOp_1:value:0ctrain/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp:value:0etrain/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/BatchNorm/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:2N
Ltrain/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/BatchNorm/FusedBatchNormV3?
Qtrain/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/BatchNorm/AssignMovingAvg/sub_1Subbtrain/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/BatchNorm/AssignMovingAvg/ReadVariableOp:value:0Ytrain/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/BatchNorm/FusedBatchNormV3:batch_mean:0*
T0*
_output_shapes	
:?2S
Qtrain/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/BatchNorm/AssignMovingAvg/sub_1?
Qtrain/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/BatchNorm/AssignMovingAvg/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2S
Qtrain/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/BatchNorm/AssignMovingAvg/sub/x?
Otrain/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/BatchNorm/AssignMovingAvg/subSubZtrain/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/BatchNorm/AssignMovingAvg/sub/x:output:0placeholder_2*
T0*
_output_shapes
: 2Q
Otrain/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/BatchNorm/AssignMovingAvg/sub?
Otrain/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/BatchNorm/AssignMovingAvg/mulMulUtrain/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/BatchNorm/AssignMovingAvg/sub_1:z:0Strain/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/BatchNorm/AssignMovingAvg/sub:z:0*
T0*
_output_shapes	
:?2Q
Otrain/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/BatchNorm/AssignMovingAvg/mul?
Ktrain/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/BatchNorm/AssignMovingAvgAssignSubVariableOpdtrain_mobilenetv1_mobilenetv1_conv2d_13_pointwise_batchnorm_fusedbatchnormv3_readvariableop_resourceStrain/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/BatchNorm/AssignMovingAvg/mul:z:0*
_output_shapes
 *
dtype02M
Ktrain/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/BatchNorm/AssignMovingAvg?
\train/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/BatchNorm/AssignMovingAvg_1/ReadVariableOpReadVariableOpftrain_mobilenetv1_mobilenetv1_conv2d_13_pointwise_batchnorm_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02^
\train/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/BatchNorm/AssignMovingAvg_1/ReadVariableOp?
Strain/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/BatchNorm/AssignMovingAvg_1/sub_1Subdtrain/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/BatchNorm/AssignMovingAvg_1/ReadVariableOp:value:0]train/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/BatchNorm/FusedBatchNormV3:batch_variance:0*
T0*
_output_shapes	
:?2U
Strain/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/BatchNorm/AssignMovingAvg_1/sub_1?
Strain/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/BatchNorm/AssignMovingAvg_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2U
Strain/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/BatchNorm/AssignMovingAvg_1/sub/x?
Qtrain/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/BatchNorm/AssignMovingAvg_1/subSub\train/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/BatchNorm/AssignMovingAvg_1/sub/x:output:0placeholder_2*
T0*
_output_shapes
: 2S
Qtrain/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/BatchNorm/AssignMovingAvg_1/sub?
Qtrain/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/BatchNorm/AssignMovingAvg_1/mulMulWtrain/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/BatchNorm/AssignMovingAvg_1/sub_1:z:0Utrain/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/BatchNorm/AssignMovingAvg_1/sub:z:0*
T0*
_output_shapes	
:?2S
Qtrain/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/BatchNorm/AssignMovingAvg_1/mul?
Mtrain/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/BatchNorm/AssignMovingAvg_1AssignSubVariableOpftrain_mobilenetv1_mobilenetv1_conv2d_13_pointwise_batchnorm_fusedbatchnormv3_readvariableop_1_resourceUtrain/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/BatchNorm/AssignMovingAvg_1/mul:z:0*
_output_shapes
 *
dtype02O
Mtrain/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/BatchNorm/AssignMovingAvg_1?
7train/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/Relu6Relu6Ptrain/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/BatchNorm/FusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,????????????????????????????29
7train/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/Relu6?
6train/MobilenetV1/Logits/global_pool/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      28
6train/MobilenetV1/Logits/global_pool/reduction_indices?
$train/MobilenetV1/Logits/global_poolMeanEtrain/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/Relu6:activations:0?train/MobilenetV1/Logits/global_pool/reduction_indices:output:0*
T0*0
_output_shapes
:??????????*
	keep_dims(2&
$train/MobilenetV1/Logits/global_pool?
1train/MobilenetV1/Logits/Dropout_1b/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *? ??23
1train/MobilenetV1/Logits/Dropout_1b/dropout/Const?
/train/MobilenetV1/Logits/Dropout_1b/dropout/MulMul-train/MobilenetV1/Logits/global_pool:output:0:train/MobilenetV1/Logits/Dropout_1b/dropout/Const:output:0*
T0*0
_output_shapes
:??????????21
/train/MobilenetV1/Logits/Dropout_1b/dropout/Mul?
1train/MobilenetV1/Logits/Dropout_1b/dropout/ShapeShape-train/MobilenetV1/Logits/global_pool:output:0*
T0*
_output_shapes
:23
1train/MobilenetV1/Logits/Dropout_1b/dropout/Shape?
Htrain/MobilenetV1/Logits/Dropout_1b/dropout/random_uniform/RandomUniformRandomUniform:train/MobilenetV1/Logits/Dropout_1b/dropout/Shape:output:0*
T0*0
_output_shapes
:??????????*
dtype02J
Htrain/MobilenetV1/Logits/Dropout_1b/dropout/random_uniform/RandomUniform?
:train/MobilenetV1/Logits/Dropout_1b/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2<
:train/MobilenetV1/Logits/Dropout_1b/dropout/GreaterEqual/y?
8train/MobilenetV1/Logits/Dropout_1b/dropout/GreaterEqualGreaterEqualQtrain/MobilenetV1/Logits/Dropout_1b/dropout/random_uniform/RandomUniform:output:0Ctrain/MobilenetV1/Logits/Dropout_1b/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:??????????2:
8train/MobilenetV1/Logits/Dropout_1b/dropout/GreaterEqual?
0train/MobilenetV1/Logits/Dropout_1b/dropout/CastCast<train/MobilenetV1/Logits/Dropout_1b/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:??????????22
0train/MobilenetV1/Logits/Dropout_1b/dropout/Cast?
1train/MobilenetV1/Logits/Dropout_1b/dropout/Mul_1Mul3train/MobilenetV1/Logits/Dropout_1b/dropout/Mul:z:04train/MobilenetV1/Logits/Dropout_1b/dropout/Cast:y:0*
T0*0
_output_shapes
:??????????23
1train/MobilenetV1/Logits/Dropout_1b/dropout/Mul_1?
<train/MobilenetV1/Logits/Conv2d_1c_1x1/Conv2D/ReadVariableOpReadVariableOpgtrain_mobilenetv1_logits_conv2d_1c_1x1_kernel_regularizer_l2_regularizer_l2loss_readvariableop_resource*(
_output_shapes
:??*
dtype02>
<train/MobilenetV1/Logits/Conv2d_1c_1x1/Conv2D/ReadVariableOp?
-train/MobilenetV1/Logits/Conv2d_1c_1x1/Conv2DConv2D5train/MobilenetV1/Logits/Dropout_1b/dropout/Mul_1:z:0Dtrain/MobilenetV1/Logits/Conv2d_1c_1x1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2/
-train/MobilenetV1/Logits/Conv2d_1c_1x1/Conv2D?
=train/MobilenetV1/Logits/Conv2d_1c_1x1/BiasAdd/ReadVariableOpReadVariableOpFtrain_mobilenetv1_logits_conv2d_1c_1x1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02?
=train/MobilenetV1/Logits/Conv2d_1c_1x1/BiasAdd/ReadVariableOp?
.train/MobilenetV1/Logits/Conv2d_1c_1x1/BiasAddBiasAdd6train/MobilenetV1/Logits/Conv2d_1c_1x1/Conv2D:output:0Etrain/MobilenetV1/Logits/Conv2d_1c_1x1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????20
.train/MobilenetV1/Logits/Conv2d_1c_1x1/BiasAdd?
'train/MobilenetV1/Logits/SpatialSqueezeSqueeze7train/MobilenetV1/Logits/Conv2d_1c_1x1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????*
squeeze_dims
2)
'train/MobilenetV1/Logits/SpatialSqueeze?
+train/MobilenetV1/Predictions/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"?????  2-
+train/MobilenetV1/Predictions/Reshape/shape?
%train/MobilenetV1/Predictions/ReshapeReshape0train/MobilenetV1/Logits/SpatialSqueeze:output:04train/MobilenetV1/Predictions/Reshape/shape:output:0*
T0*(
_output_shapes
:??????????2'
%train/MobilenetV1/Predictions/Reshape?
%train/MobilenetV1/Predictions/SoftmaxSoftmax.train/MobilenetV1/Predictions/Reshape:output:0*
T0*(
_output_shapes
:??????????2'
%train/MobilenetV1/Predictions/Softmax?
#train/MobilenetV1/Predictions/ShapeShape0train/MobilenetV1/Logits/SpatialSqueeze:output:0*
T0*
_output_shapes
:2%
#train/MobilenetV1/Predictions/Shape?
'train/MobilenetV1/Predictions/Reshape_1Reshape/train/MobilenetV1/Predictions/Softmax:softmax:0,train/MobilenetV1/Predictions/Shape:output:0*
T0*(
_output_shapes
:??????????2)
'train/MobilenetV1/Predictions/Reshape_1"U
$train_mobilenetv1_logits_global_pool-train/MobilenetV1/Logits/global_pool:output:0"[
'train_mobilenetv1_logits_spatialsqueeze0train/MobilenetV1/Logits/SpatialSqueeze:output:0"]
)train_mobilenetv1_logits_spatialsqueeze_00train/MobilenetV1/Logits/SpatialSqueeze:output:0"j
,train_mobilenetv1_mobilenetv1_conv2d_0_relu6:train/MobilenetV1/MobilenetV1/Conv2d_0/Relu6:activations:0"?
7train_mobilenetv1_mobilenetv1_conv2d_10_depthwise_relu6Etrain/MobilenetV1/MobilenetV1/Conv2d_10_depthwise/Relu6:activations:0"?
7train_mobilenetv1_mobilenetv1_conv2d_10_pointwise_relu6Etrain/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/Relu6:activations:0"?
7train_mobilenetv1_mobilenetv1_conv2d_11_depthwise_relu6Etrain/MobilenetV1/MobilenetV1/Conv2d_11_depthwise/Relu6:activations:0"?
7train_mobilenetv1_mobilenetv1_conv2d_11_pointwise_relu6Etrain/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/Relu6:activations:0"?
7train_mobilenetv1_mobilenetv1_conv2d_12_depthwise_relu6Etrain/MobilenetV1/MobilenetV1/Conv2d_12_depthwise/Relu6:activations:0"?
7train_mobilenetv1_mobilenetv1_conv2d_12_pointwise_relu6Etrain/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/Relu6:activations:0"?
7train_mobilenetv1_mobilenetv1_conv2d_13_depthwise_relu6Etrain/MobilenetV1/MobilenetV1/Conv2d_13_depthwise/Relu6:activations:0"?
7train_mobilenetv1_mobilenetv1_conv2d_13_pointwise_relu6Etrain/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/Relu6:activations:0"~
6train_mobilenetv1_mobilenetv1_conv2d_1_depthwise_relu6Dtrain/MobilenetV1/MobilenetV1/Conv2d_1_depthwise/Relu6:activations:0"~
6train_mobilenetv1_mobilenetv1_conv2d_1_pointwise_relu6Dtrain/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/Relu6:activations:0"~
6train_mobilenetv1_mobilenetv1_conv2d_2_depthwise_relu6Dtrain/MobilenetV1/MobilenetV1/Conv2d_2_depthwise/Relu6:activations:0"~
6train_mobilenetv1_mobilenetv1_conv2d_2_pointwise_relu6Dtrain/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/Relu6:activations:0"~
6train_mobilenetv1_mobilenetv1_conv2d_3_depthwise_relu6Dtrain/MobilenetV1/MobilenetV1/Conv2d_3_depthwise/Relu6:activations:0"~
6train_mobilenetv1_mobilenetv1_conv2d_3_pointwise_relu6Dtrain/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/Relu6:activations:0"~
6train_mobilenetv1_mobilenetv1_conv2d_4_depthwise_relu6Dtrain/MobilenetV1/MobilenetV1/Conv2d_4_depthwise/Relu6:activations:0"~
6train_mobilenetv1_mobilenetv1_conv2d_4_pointwise_relu6Dtrain/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/Relu6:activations:0"~
6train_mobilenetv1_mobilenetv1_conv2d_5_depthwise_relu6Dtrain/MobilenetV1/MobilenetV1/Conv2d_5_depthwise/Relu6:activations:0"~
6train_mobilenetv1_mobilenetv1_conv2d_5_pointwise_relu6Dtrain/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/Relu6:activations:0"~
6train_mobilenetv1_mobilenetv1_conv2d_6_depthwise_relu6Dtrain/MobilenetV1/MobilenetV1/Conv2d_6_depthwise/Relu6:activations:0"~
6train_mobilenetv1_mobilenetv1_conv2d_6_pointwise_relu6Dtrain/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/Relu6:activations:0"~
6train_mobilenetv1_mobilenetv1_conv2d_7_depthwise_relu6Dtrain/MobilenetV1/MobilenetV1/Conv2d_7_depthwise/Relu6:activations:0"~
6train_mobilenetv1_mobilenetv1_conv2d_7_pointwise_relu6Dtrain/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/Relu6:activations:0"~
6train_mobilenetv1_mobilenetv1_conv2d_8_depthwise_relu6Dtrain/MobilenetV1/MobilenetV1/Conv2d_8_depthwise/Relu6:activations:0"~
6train_mobilenetv1_mobilenetv1_conv2d_8_pointwise_relu6Dtrain/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/Relu6:activations:0"~
6train_mobilenetv1_mobilenetv1_conv2d_9_depthwise_relu6Dtrain/MobilenetV1/MobilenetV1/Conv2d_9_depthwise/Relu6:activations:0"~
6train_mobilenetv1_mobilenetv1_conv2d_9_pointwise_relu6Dtrain/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/Relu6:activations:0"[
'train_mobilenetv1_predictions_reshape_10train/MobilenetV1/Predictions/Reshape_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:+???????????????????????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2?
@train/MobilenetV1/MobilenetV1/Conv2d_0/BatchNorm/AssignMovingAvg@train/MobilenetV1/MobilenetV1/Conv2d_0/BatchNorm/AssignMovingAvg2?
Btrain/MobilenetV1/MobilenetV1/Conv2d_0/BatchNorm/AssignMovingAvg_1Btrain/MobilenetV1/MobilenetV1/Conv2d_0/BatchNorm/AssignMovingAvg_12?
Ktrain/MobilenetV1/MobilenetV1/Conv2d_10_depthwise/BatchNorm/AssignMovingAvgKtrain/MobilenetV1/MobilenetV1/Conv2d_10_depthwise/BatchNorm/AssignMovingAvg2?
Mtrain/MobilenetV1/MobilenetV1/Conv2d_10_depthwise/BatchNorm/AssignMovingAvg_1Mtrain/MobilenetV1/MobilenetV1/Conv2d_10_depthwise/BatchNorm/AssignMovingAvg_12?
Ktrain/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/BatchNorm/AssignMovingAvgKtrain/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/BatchNorm/AssignMovingAvg2?
Mtrain/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/BatchNorm/AssignMovingAvg_1Mtrain/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/BatchNorm/AssignMovingAvg_12?
Ktrain/MobilenetV1/MobilenetV1/Conv2d_11_depthwise/BatchNorm/AssignMovingAvgKtrain/MobilenetV1/MobilenetV1/Conv2d_11_depthwise/BatchNorm/AssignMovingAvg2?
Mtrain/MobilenetV1/MobilenetV1/Conv2d_11_depthwise/BatchNorm/AssignMovingAvg_1Mtrain/MobilenetV1/MobilenetV1/Conv2d_11_depthwise/BatchNorm/AssignMovingAvg_12?
Ktrain/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/BatchNorm/AssignMovingAvgKtrain/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/BatchNorm/AssignMovingAvg2?
Mtrain/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/BatchNorm/AssignMovingAvg_1Mtrain/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/BatchNorm/AssignMovingAvg_12?
Ktrain/MobilenetV1/MobilenetV1/Conv2d_12_depthwise/BatchNorm/AssignMovingAvgKtrain/MobilenetV1/MobilenetV1/Conv2d_12_depthwise/BatchNorm/AssignMovingAvg2?
Mtrain/MobilenetV1/MobilenetV1/Conv2d_12_depthwise/BatchNorm/AssignMovingAvg_1Mtrain/MobilenetV1/MobilenetV1/Conv2d_12_depthwise/BatchNorm/AssignMovingAvg_12?
Ktrain/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/BatchNorm/AssignMovingAvgKtrain/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/BatchNorm/AssignMovingAvg2?
Mtrain/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/BatchNorm/AssignMovingAvg_1Mtrain/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/BatchNorm/AssignMovingAvg_12?
Ktrain/MobilenetV1/MobilenetV1/Conv2d_13_depthwise/BatchNorm/AssignMovingAvgKtrain/MobilenetV1/MobilenetV1/Conv2d_13_depthwise/BatchNorm/AssignMovingAvg2?
Mtrain/MobilenetV1/MobilenetV1/Conv2d_13_depthwise/BatchNorm/AssignMovingAvg_1Mtrain/MobilenetV1/MobilenetV1/Conv2d_13_depthwise/BatchNorm/AssignMovingAvg_12?
Ktrain/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/BatchNorm/AssignMovingAvgKtrain/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/BatchNorm/AssignMovingAvg2?
Mtrain/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/BatchNorm/AssignMovingAvg_1Mtrain/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/BatchNorm/AssignMovingAvg_12?
Jtrain/MobilenetV1/MobilenetV1/Conv2d_1_depthwise/BatchNorm/AssignMovingAvgJtrain/MobilenetV1/MobilenetV1/Conv2d_1_depthwise/BatchNorm/AssignMovingAvg2?
Ltrain/MobilenetV1/MobilenetV1/Conv2d_1_depthwise/BatchNorm/AssignMovingAvg_1Ltrain/MobilenetV1/MobilenetV1/Conv2d_1_depthwise/BatchNorm/AssignMovingAvg_12?
Jtrain/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/BatchNorm/AssignMovingAvgJtrain/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/BatchNorm/AssignMovingAvg2?
Ltrain/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/BatchNorm/AssignMovingAvg_1Ltrain/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/BatchNorm/AssignMovingAvg_12?
Jtrain/MobilenetV1/MobilenetV1/Conv2d_2_depthwise/BatchNorm/AssignMovingAvgJtrain/MobilenetV1/MobilenetV1/Conv2d_2_depthwise/BatchNorm/AssignMovingAvg2?
Ltrain/MobilenetV1/MobilenetV1/Conv2d_2_depthwise/BatchNorm/AssignMovingAvg_1Ltrain/MobilenetV1/MobilenetV1/Conv2d_2_depthwise/BatchNorm/AssignMovingAvg_12?
Jtrain/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/BatchNorm/AssignMovingAvgJtrain/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/BatchNorm/AssignMovingAvg2?
Ltrain/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/BatchNorm/AssignMovingAvg_1Ltrain/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/BatchNorm/AssignMovingAvg_12?
Jtrain/MobilenetV1/MobilenetV1/Conv2d_3_depthwise/BatchNorm/AssignMovingAvgJtrain/MobilenetV1/MobilenetV1/Conv2d_3_depthwise/BatchNorm/AssignMovingAvg2?
Ltrain/MobilenetV1/MobilenetV1/Conv2d_3_depthwise/BatchNorm/AssignMovingAvg_1Ltrain/MobilenetV1/MobilenetV1/Conv2d_3_depthwise/BatchNorm/AssignMovingAvg_12?
Jtrain/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/BatchNorm/AssignMovingAvgJtrain/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/BatchNorm/AssignMovingAvg2?
Ltrain/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/BatchNorm/AssignMovingAvg_1Ltrain/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/BatchNorm/AssignMovingAvg_12?
Jtrain/MobilenetV1/MobilenetV1/Conv2d_4_depthwise/BatchNorm/AssignMovingAvgJtrain/MobilenetV1/MobilenetV1/Conv2d_4_depthwise/BatchNorm/AssignMovingAvg2?
Ltrain/MobilenetV1/MobilenetV1/Conv2d_4_depthwise/BatchNorm/AssignMovingAvg_1Ltrain/MobilenetV1/MobilenetV1/Conv2d_4_depthwise/BatchNorm/AssignMovingAvg_12?
Jtrain/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/BatchNorm/AssignMovingAvgJtrain/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/BatchNorm/AssignMovingAvg2?
Ltrain/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/BatchNorm/AssignMovingAvg_1Ltrain/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/BatchNorm/AssignMovingAvg_12?
Jtrain/MobilenetV1/MobilenetV1/Conv2d_5_depthwise/BatchNorm/AssignMovingAvgJtrain/MobilenetV1/MobilenetV1/Conv2d_5_depthwise/BatchNorm/AssignMovingAvg2?
Ltrain/MobilenetV1/MobilenetV1/Conv2d_5_depthwise/BatchNorm/AssignMovingAvg_1Ltrain/MobilenetV1/MobilenetV1/Conv2d_5_depthwise/BatchNorm/AssignMovingAvg_12?
Jtrain/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/BatchNorm/AssignMovingAvgJtrain/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/BatchNorm/AssignMovingAvg2?
Ltrain/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/BatchNorm/AssignMovingAvg_1Ltrain/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/BatchNorm/AssignMovingAvg_12?
Jtrain/MobilenetV1/MobilenetV1/Conv2d_6_depthwise/BatchNorm/AssignMovingAvgJtrain/MobilenetV1/MobilenetV1/Conv2d_6_depthwise/BatchNorm/AssignMovingAvg2?
Ltrain/MobilenetV1/MobilenetV1/Conv2d_6_depthwise/BatchNorm/AssignMovingAvg_1Ltrain/MobilenetV1/MobilenetV1/Conv2d_6_depthwise/BatchNorm/AssignMovingAvg_12?
Jtrain/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/BatchNorm/AssignMovingAvgJtrain/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/BatchNorm/AssignMovingAvg2?
Ltrain/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/BatchNorm/AssignMovingAvg_1Ltrain/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/BatchNorm/AssignMovingAvg_12?
Jtrain/MobilenetV1/MobilenetV1/Conv2d_7_depthwise/BatchNorm/AssignMovingAvgJtrain/MobilenetV1/MobilenetV1/Conv2d_7_depthwise/BatchNorm/AssignMovingAvg2?
Ltrain/MobilenetV1/MobilenetV1/Conv2d_7_depthwise/BatchNorm/AssignMovingAvg_1Ltrain/MobilenetV1/MobilenetV1/Conv2d_7_depthwise/BatchNorm/AssignMovingAvg_12?
Jtrain/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/BatchNorm/AssignMovingAvgJtrain/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/BatchNorm/AssignMovingAvg2?
Ltrain/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/BatchNorm/AssignMovingAvg_1Ltrain/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/BatchNorm/AssignMovingAvg_12?
Jtrain/MobilenetV1/MobilenetV1/Conv2d_8_depthwise/BatchNorm/AssignMovingAvgJtrain/MobilenetV1/MobilenetV1/Conv2d_8_depthwise/BatchNorm/AssignMovingAvg2?
Ltrain/MobilenetV1/MobilenetV1/Conv2d_8_depthwise/BatchNorm/AssignMovingAvg_1Ltrain/MobilenetV1/MobilenetV1/Conv2d_8_depthwise/BatchNorm/AssignMovingAvg_12?
Jtrain/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/BatchNorm/AssignMovingAvgJtrain/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/BatchNorm/AssignMovingAvg2?
Ltrain/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/BatchNorm/AssignMovingAvg_1Ltrain/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/BatchNorm/AssignMovingAvg_12?
Jtrain/MobilenetV1/MobilenetV1/Conv2d_9_depthwise/BatchNorm/AssignMovingAvgJtrain/MobilenetV1/MobilenetV1/Conv2d_9_depthwise/BatchNorm/AssignMovingAvg2?
Ltrain/MobilenetV1/MobilenetV1/Conv2d_9_depthwise/BatchNorm/AssignMovingAvg_1Ltrain/MobilenetV1/MobilenetV1/Conv2d_9_depthwise/BatchNorm/AssignMovingAvg_12?
Jtrain/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/BatchNorm/AssignMovingAvgJtrain/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/BatchNorm/AssignMovingAvg2?
Ltrain/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/BatchNorm/AssignMovingAvg_1Ltrain/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/BatchNorm/AssignMovingAvg_1:G C
A
_output_shapes/
-:+???????????????????????????:

_output_shapes
: 
?C
? 
__inference_call_fn_7068

inputs
batch_norm_momentum!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:#
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:#
	unknown_9:0

unknown_10:0

unknown_11:0

unknown_12:0

unknown_13:0$

unknown_14:0

unknown_15:0

unknown_16:0

unknown_17:0

unknown_18:0$

unknown_19:0`

unknown_20:`

unknown_21:`

unknown_22:`

unknown_23:`$

unknown_24:`

unknown_25:`

unknown_26:`

unknown_27:`

unknown_28:`$

unknown_29:``

unknown_30:`

unknown_31:`

unknown_32:`

unknown_33:`$

unknown_34:`

unknown_35:`

unknown_36:`

unknown_37:`

unknown_38:`%

unknown_39:`?

unknown_40:	?

unknown_41:	?

unknown_42:	?

unknown_43:	?%

unknown_44:?

unknown_45:	?

unknown_46:	?

unknown_47:	?

unknown_48:	?&

unknown_49:??

unknown_50:	?

unknown_51:	?

unknown_52:	?

unknown_53:	?%

unknown_54:?

unknown_55:	?

unknown_56:	?

unknown_57:	?

unknown_58:	?&

unknown_59:??

unknown_60:	?

unknown_61:	?

unknown_62:	?

unknown_63:	?%

unknown_64:?

unknown_65:	?

unknown_66:	?

unknown_67:	?

unknown_68:	?&

unknown_69:??

unknown_70:	?

unknown_71:	?

unknown_72:	?

unknown_73:	?%

unknown_74:?

unknown_75:	?

unknown_76:	?

unknown_77:	?

unknown_78:	?&

unknown_79:??

unknown_80:	?

unknown_81:	?

unknown_82:	?

unknown_83:	?%

unknown_84:?

unknown_85:	?

unknown_86:	?

unknown_87:	?

unknown_88:	?&

unknown_89:??

unknown_90:	?

unknown_91:	?

unknown_92:	?

unknown_93:	?%

unknown_94:?

unknown_95:	?

unknown_96:	?

unknown_97:	?

unknown_98:	?&

unknown_99:??
unknown_100:	?
unknown_101:	?
unknown_102:	?
unknown_103:	?&
unknown_104:?
unknown_105:	?
unknown_106:	?
unknown_107:	?
unknown_108:	?'
unknown_109:??
unknown_110:	?
unknown_111:	?
unknown_112:	?
unknown_113:	?&
unknown_114:?
unknown_115:	?
unknown_116:	?
unknown_117:	?
unknown_118:	?'
unknown_119:??
unknown_120:	?
unknown_121:	?
unknown_122:	?
unknown_123:	?&
unknown_124:?
unknown_125:	?
unknown_126:	?
unknown_127:	?
unknown_128:	?'
unknown_129:??
unknown_130:	?
unknown_131:	?
unknown_132:	?
unknown_133:	?'
unknown_134:??
unknown_135:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_norm_momentumunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58
unknown_59
unknown_60
unknown_61
unknown_62
unknown_63
unknown_64
unknown_65
unknown_66
unknown_67
unknown_68
unknown_69
unknown_70
unknown_71
unknown_72
unknown_73
unknown_74
unknown_75
unknown_76
unknown_77
unknown_78
unknown_79
unknown_80
unknown_81
unknown_82
unknown_83
unknown_84
unknown_85
unknown_86
unknown_87
unknown_88
unknown_89
unknown_90
unknown_91
unknown_92
unknown_93
unknown_94
unknown_95
unknown_96
unknown_97
unknown_98
unknown_99unknown_100unknown_101unknown_102unknown_103unknown_104unknown_105unknown_106unknown_107unknown_108unknown_109unknown_110unknown_111unknown_112unknown_113unknown_114unknown_115unknown_116unknown_117unknown_118unknown_119unknown_120unknown_121unknown_122unknown_123unknown_124unknown_125unknown_126unknown_127unknown_128unknown_129unknown_130unknown_131unknown_132unknown_133unknown_134unknown_135*?
Tin?
?2?*+
Tout#
!2*?

_output_shapes?

?
:??????????:+???????????????????????????:,????????????????????????????:,????????????????????????????:,????????????????????????????:,????????????????????????????:,????????????????????????????:,????????????????????????????:,????????????????????????????:,????????????????????????????:+???????????????????????????:+???????????????????????????0:+???????????????????????????0:+???????????????????????????`:+???????????????????????????`:+???????????????????????????`:+???????????????????????????`:,????????????????????????????:,????????????????????????????:,????????????????????????????:,????????????????????????????:,????????????????????????????:,????????????????????????????:,????????????????????????????:,????????????????????????????:,????????????????????????????:,????????????????????????????:,????????????????????????????:??????????:??????????:??????????*?
_read_only_resource_inputs?
??	
 !"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~???????????*-
config_proto

CPU

GPU 2J 8? * 
fR
__inference_pruned_33932
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

IdentityD
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:+???????????????????????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs:KG

_output_shapes
: 
-
_user_specified_namebatch_norm_momentum
??
?v
 __inference__traced_restore_8208
file_prefix'
assignvariableop_save_counter:	 Y
Kassignvariableop_1_mobilenetv1_conv2d_1_depthwise_batchnorm_moving_variance:O
Aassignvariableop_2_mobilenetv1_conv2d_3_depthwise_batchnorm_gamma:`P
Aassignvariableop_3_mobilenetv1_conv2d_7_pointwise_batchnorm_gamma:	?O
@assignvariableop_4_mobilenetv1_conv2d_8_pointwise_batchnorm_beta:	?P
Aassignvariableop_5_mobilenetv1_conv2d_11_pointwise_batchnorm_beta:	?]
Cassignvariableop_6_mobilenetv1_conv2d_4_depthwise_depthwise_weights:`W
;assignvariableop_7_mobilenetv1_logits_conv2d_1c_1x1_weights:??K
=assignvariableop_8_mobilenetv1_conv2d_0_batchnorm_moving_mean:U
Gassignvariableop_9_mobilenetv1_conv2d_3_pointwise_batchnorm_moving_mean:`V
Hassignvariableop_10_mobilenetv1_conv2d_4_depthwise_batchnorm_moving_mean:`[
Lassignvariableop_11_mobilenetv1_conv2d_5_depthwise_batchnorm_moving_variance:	?Q
Bassignvariableop_12_mobilenetv1_conv2d_6_depthwise_batchnorm_gamma:	?[
Lassignvariableop_13_mobilenetv1_conv2d_7_pointwise_batchnorm_moving_variance:	?[
Lassignvariableop_14_mobilenetv1_conv2d_9_depthwise_batchnorm_moving_variance:	?[
Lassignvariableop_15_mobilenetv1_conv2d_9_pointwise_batchnorm_moving_variance:	?W
;assignvariableop_16_mobilenetv1_conv2d_11_pointwise_weights:??V
Hassignvariableop_17_mobilenetv1_conv2d_1_pointwise_batchnorm_moving_mean:0W
Hassignvariableop_18_mobilenetv1_conv2d_5_depthwise_batchnorm_moving_mean:	?X
Iassignvariableop_19_mobilenetv1_conv2d_10_pointwise_batchnorm_moving_mean:	?P
Aassignvariableop_20_mobilenetv1_conv2d_7_pointwise_batchnorm_beta:	?J
0assignvariableop_21_mobilenetv1_conv2d_0_weights:Z
Lassignvariableop_22_mobilenetv1_conv2d_2_depthwise_batchnorm_moving_variance:0Z
Lassignvariableop_23_mobilenetv1_conv2d_4_depthwise_batchnorm_moving_variance:`W
Hassignvariableop_24_mobilenetv1_conv2d_6_pointwise_batchnorm_moving_mean:	?P
Aassignvariableop_25_mobilenetv1_conv2d_9_depthwise_batchnorm_beta:	?R
Cassignvariableop_26_mobilenetv1_conv2d_11_depthwise_batchnorm_gamma:	?V
:assignvariableop_27_mobilenetv1_conv2d_6_pointwise_weights:??P
Aassignvariableop_28_mobilenetv1_conv2d_6_pointwise_batchnorm_beta:	?O
Aassignvariableop_29_mobilenetv1_conv2d_1_depthwise_batchnorm_beta:P
Bassignvariableop_30_mobilenetv1_conv2d_3_pointwise_batchnorm_gamma:`Q
Bassignvariableop_31_mobilenetv1_conv2d_10_depthwise_batchnorm_beta:	?J
;assignvariableop_32_mobilenetv1_logits_conv2d_1c_1x1_biases:	?W
Hassignvariableop_33_mobilenetv1_conv2d_8_pointwise_batchnorm_moving_mean:	?P
Aassignvariableop_34_mobilenetv1_conv2d_5_pointwise_batchnorm_beta:	?^
Dassignvariableop_35_mobilenetv1_conv2d_2_depthwise_depthwise_weights:0T
:assignvariableop_36_mobilenetv1_conv2d_1_pointwise_weights:0P
Bassignvariableop_37_mobilenetv1_conv2d_1_pointwise_batchnorm_gamma:0P
Aassignvariableop_38_mobilenetv1_conv2d_5_depthwise_batchnorm_beta:	?[
Lassignvariableop_39_mobilenetv1_conv2d_5_pointwise_batchnorm_moving_variance:	?Q
Bassignvariableop_40_mobilenetv1_conv2d_8_depthwise_batchnorm_gamma:	?`
Eassignvariableop_41_mobilenetv1_conv2d_13_depthwise_depthwise_weights:?[
Lassignvariableop_42_mobilenetv1_conv2d_6_depthwise_batchnorm_moving_variance:	?V
:assignvariableop_43_mobilenetv1_conv2d_9_pointwise_weights:??\
Massignvariableop_44_mobilenetv1_conv2d_10_depthwise_batchnorm_moving_variance:	?Q
Bassignvariableop_45_mobilenetv1_conv2d_12_pointwise_batchnorm_beta:	?O
Aassignvariableop_46_mobilenetv1_conv2d_2_depthwise_batchnorm_beta:0V
Hassignvariableop_47_mobilenetv1_conv2d_2_pointwise_batchnorm_moving_mean:`O
Aassignvariableop_48_mobilenetv1_conv2d_3_pointwise_batchnorm_beta:`W
Hassignvariableop_49_mobilenetv1_conv2d_6_depthwise_batchnorm_moving_mean:	?R
Cassignvariableop_50_mobilenetv1_conv2d_10_pointwise_batchnorm_gamma:	?R
Cassignvariableop_51_mobilenetv1_conv2d_12_depthwise_batchnorm_gamma:	?X
Iassignvariableop_52_mobilenetv1_conv2d_13_depthwise_batchnorm_moving_mean:	?_
Dassignvariableop_53_mobilenetv1_conv2d_5_depthwise_depthwise_weights:?Q
Bassignvariableop_54_mobilenetv1_conv2d_13_depthwise_batchnorm_beta:	?Q
Bassignvariableop_55_mobilenetv1_conv2d_8_pointwise_batchnorm_gamma:	?O
Aassignvariableop_56_mobilenetv1_conv2d_3_depthwise_batchnorm_beta:`Q
Bassignvariableop_57_mobilenetv1_conv2d_7_depthwise_batchnorm_gamma:	?[
Lassignvariableop_58_mobilenetv1_conv2d_8_depthwise_batchnorm_moving_variance:	?X
Iassignvariableop_59_mobilenetv1_conv2d_10_depthwise_batchnorm_moving_mean:	?\
Massignvariableop_60_mobilenetv1_conv2d_10_pointwise_batchnorm_moving_variance:	?W
;assignvariableop_61_mobilenetv1_conv2d_12_pointwise_weights:??\
Massignvariableop_62_mobilenetv1_conv2d_12_pointwise_batchnorm_moving_variance:	?P
Bassignvariableop_63_mobilenetv1_conv2d_4_depthwise_batchnorm_gamma:`W
Hassignvariableop_64_mobilenetv1_conv2d_8_depthwise_batchnorm_moving_mean:	?X
Iassignvariableop_65_mobilenetv1_conv2d_13_pointwise_batchnorm_moving_mean:	?[
Lassignvariableop_66_mobilenetv1_conv2d_7_depthwise_batchnorm_moving_variance:	?V
:assignvariableop_67_mobilenetv1_conv2d_7_pointwise_weights:??V
:assignvariableop_68_mobilenetv1_conv2d_5_pointwise_weights:??Z
Lassignvariableop_69_mobilenetv1_conv2d_1_pointwise_batchnorm_moving_variance:0V
Hassignvariableop_70_mobilenetv1_conv2d_1_depthwise_batchnorm_moving_mean:Z
Lassignvariableop_71_mobilenetv1_conv2d_3_pointwise_batchnorm_moving_variance:`W
Hassignvariableop_72_mobilenetv1_conv2d_9_pointwise_batchnorm_moving_mean:	?Q
Bassignvariableop_73_mobilenetv1_conv2d_10_pointwise_batchnorm_beta:	?\
Massignvariableop_74_mobilenetv1_conv2d_12_depthwise_batchnorm_moving_variance:	?W
Hassignvariableop_75_mobilenetv1_conv2d_4_pointwise_batchnorm_moving_mean:	?W
;assignvariableop_76_mobilenetv1_conv2d_10_pointwise_weights:??^
Dassignvariableop_77_mobilenetv1_conv2d_1_depthwise_depthwise_weights:O
Aassignvariableop_78_mobilenetv1_conv2d_1_pointwise_batchnorm_beta:0P
Bassignvariableop_79_mobilenetv1_conv2d_0_batchnorm_moving_variance:P
Bassignvariableop_80_mobilenetv1_conv2d_2_pointwise_batchnorm_gamma:`_
Dassignvariableop_81_mobilenetv1_conv2d_9_depthwise_depthwise_weights:?X
Iassignvariableop_82_mobilenetv1_conv2d_11_pointwise_batchnorm_moving_mean:	?_
Dassignvariableop_83_mobilenetv1_conv2d_8_depthwise_depthwise_weights:?P
Aassignvariableop_84_mobilenetv1_conv2d_8_depthwise_batchnorm_beta:	?[
Lassignvariableop_85_mobilenetv1_conv2d_8_pointwise_batchnorm_moving_variance:	?T
:assignvariableop_86_mobilenetv1_conv2d_3_pointwise_weights:``Q
Bassignvariableop_87_mobilenetv1_conv2d_12_depthwise_batchnorm_beta:	?\
Massignvariableop_88_mobilenetv1_conv2d_13_depthwise_batchnorm_moving_variance:	?W
;assignvariableop_89_mobilenetv1_conv2d_13_pointwise_weights:??_
Dassignvariableop_90_mobilenetv1_conv2d_7_depthwise_depthwise_weights:?P
Aassignvariableop_91_mobilenetv1_conv2d_7_depthwise_batchnorm_beta:	?R
Cassignvariableop_92_mobilenetv1_conv2d_13_pointwise_batchnorm_gamma:	?F
8assignvariableop_93_mobilenetv1_conv2d_0_batchnorm_gamma:O
Aassignvariableop_94_mobilenetv1_conv2d_4_depthwise_batchnorm_beta:`P
Aassignvariableop_95_mobilenetv1_conv2d_9_pointwise_batchnorm_beta:	?Q
Bassignvariableop_96_mobilenetv1_conv2d_5_pointwise_batchnorm_gamma:	?P
Bassignvariableop_97_mobilenetv1_conv2d_2_depthwise_batchnorm_gamma:0Q
Bassignvariableop_98_mobilenetv1_conv2d_4_pointwise_batchnorm_gamma:	?_
Dassignvariableop_99_mobilenetv1_conv2d_6_depthwise_depthwise_weights:?R
Cassignvariableop_100_mobilenetv1_conv2d_9_depthwise_batchnorm_gamma:	?]
Nassignvariableop_101_mobilenetv1_conv2d_11_depthwise_batchnorm_moving_variance:	?S
Dassignvariableop_102_mobilenetv1_conv2d_13_depthwise_batchnorm_gamma:	?]
Nassignvariableop_103_mobilenetv1_conv2d_13_pointwise_batchnorm_moving_variance:	?Y
Jassignvariableop_104_mobilenetv1_conv2d_11_depthwise_batchnorm_moving_mean:	?W
;assignvariableop_105_mobilenetv1_conv2d_8_pointwise_weights:??W
Iassignvariableop_106_mobilenetv1_conv2d_3_depthwise_batchnorm_moving_mean:`V
;assignvariableop_107_mobilenetv1_conv2d_4_pointwise_weights:`?\
Massignvariableop_108_mobilenetv1_conv2d_4_pointwise_batchnorm_moving_variance:	?Q
Bassignvariableop_109_mobilenetv1_conv2d_6_depthwise_batchnorm_beta:	?\
Massignvariableop_110_mobilenetv1_conv2d_6_pointwise_batchnorm_moving_variance:	?X
Iassignvariableop_111_mobilenetv1_conv2d_9_depthwise_batchnorm_moving_mean:	?R
Cassignvariableop_112_mobilenetv1_conv2d_9_pointwise_batchnorm_gamma:	?Y
Jassignvariableop_113_mobilenetv1_conv2d_12_pointwise_batchnorm_moving_mean:	?R
Cassignvariableop_114_mobilenetv1_conv2d_13_pointwise_batchnorm_beta:	?X
Iassignvariableop_115_mobilenetv1_conv2d_7_pointwise_batchnorm_moving_mean:	?Y
Jassignvariableop_116_mobilenetv1_conv2d_12_depthwise_batchnorm_moving_mean:	?_
Eassignvariableop_117_mobilenetv1_conv2d_3_depthwise_depthwise_weights:`Q
Bassignvariableop_118_mobilenetv1_conv2d_4_pointwise_batchnorm_beta:	?W
Iassignvariableop_119_mobilenetv1_conv2d_2_depthwise_batchnorm_moving_mean:0P
Bassignvariableop_120_mobilenetv1_conv2d_2_pointwise_batchnorm_beta:`S
Dassignvariableop_121_mobilenetv1_conv2d_10_depthwise_batchnorm_gamma:	?a
Fassignvariableop_122_mobilenetv1_conv2d_12_depthwise_depthwise_weights:?F
8assignvariableop_123_mobilenetv1_conv2d_0_batchnorm_beta:a
Fassignvariableop_124_mobilenetv1_conv2d_11_depthwise_depthwise_weights:?R
Cassignvariableop_125_mobilenetv1_conv2d_11_depthwise_batchnorm_beta:	?]
Nassignvariableop_126_mobilenetv1_conv2d_11_pointwise_batchnorm_moving_variance:	?X
Iassignvariableop_127_mobilenetv1_conv2d_7_depthwise_batchnorm_moving_mean:	?Q
Cassignvariableop_128_mobilenetv1_conv2d_1_depthwise_batchnorm_gamma:[
Massignvariableop_129_mobilenetv1_conv2d_3_depthwise_batchnorm_moving_variance:`X
Iassignvariableop_130_mobilenetv1_conv2d_5_pointwise_batchnorm_moving_mean:	?S
Dassignvariableop_131_mobilenetv1_conv2d_12_pointwise_batchnorm_gamma:	?U
;assignvariableop_132_mobilenetv1_conv2d_2_pointwise_weights:0`[
Massignvariableop_133_mobilenetv1_conv2d_2_pointwise_batchnorm_moving_variance:`R
Cassignvariableop_134_mobilenetv1_conv2d_5_depthwise_batchnorm_gamma:	?a
Fassignvariableop_135_mobilenetv1_conv2d_10_depthwise_depthwise_weights:?S
Dassignvariableop_136_mobilenetv1_conv2d_11_pointwise_batchnorm_gamma:	?R
Cassignvariableop_137_mobilenetv1_conv2d_6_pointwise_batchnorm_gamma:	?
identity_139??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_100?AssignVariableOp_101?AssignVariableOp_102?AssignVariableOp_103?AssignVariableOp_104?AssignVariableOp_105?AssignVariableOp_106?AssignVariableOp_107?AssignVariableOp_108?AssignVariableOp_109?AssignVariableOp_11?AssignVariableOp_110?AssignVariableOp_111?AssignVariableOp_112?AssignVariableOp_113?AssignVariableOp_114?AssignVariableOp_115?AssignVariableOp_116?AssignVariableOp_117?AssignVariableOp_118?AssignVariableOp_119?AssignVariableOp_12?AssignVariableOp_120?AssignVariableOp_121?AssignVariableOp_122?AssignVariableOp_123?AssignVariableOp_124?AssignVariableOp_125?AssignVariableOp_126?AssignVariableOp_127?AssignVariableOp_128?AssignVariableOp_129?AssignVariableOp_13?AssignVariableOp_130?AssignVariableOp_131?AssignVariableOp_132?AssignVariableOp_133?AssignVariableOp_134?AssignVariableOp_135?AssignVariableOp_136?AssignVariableOp_137?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_62?AssignVariableOp_63?AssignVariableOp_64?AssignVariableOp_65?AssignVariableOp_66?AssignVariableOp_67?AssignVariableOp_68?AssignVariableOp_69?AssignVariableOp_7?AssignVariableOp_70?AssignVariableOp_71?AssignVariableOp_72?AssignVariableOp_73?AssignVariableOp_74?AssignVariableOp_75?AssignVariableOp_76?AssignVariableOp_77?AssignVariableOp_78?AssignVariableOp_79?AssignVariableOp_8?AssignVariableOp_80?AssignVariableOp_81?AssignVariableOp_82?AssignVariableOp_83?AssignVariableOp_84?AssignVariableOp_85?AssignVariableOp_86?AssignVariableOp_87?AssignVariableOp_88?AssignVariableOp_89?AssignVariableOp_9?AssignVariableOp_90?AssignVariableOp_91?AssignVariableOp_92?AssignVariableOp_93?AssignVariableOp_94?AssignVariableOp_95?AssignVariableOp_96?AssignVariableOp_97?AssignVariableOp_98?AssignVariableOp_99?-
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes	
:?*
dtype0*?,
value?,B?,?B'save_counter/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB'variables/32/.ATTRIBUTES/VARIABLE_VALUEB'variables/33/.ATTRIBUTES/VARIABLE_VALUEB'variables/34/.ATTRIBUTES/VARIABLE_VALUEB'variables/35/.ATTRIBUTES/VARIABLE_VALUEB'variables/36/.ATTRIBUTES/VARIABLE_VALUEB'variables/37/.ATTRIBUTES/VARIABLE_VALUEB'variables/38/.ATTRIBUTES/VARIABLE_VALUEB'variables/39/.ATTRIBUTES/VARIABLE_VALUEB'variables/40/.ATTRIBUTES/VARIABLE_VALUEB'variables/41/.ATTRIBUTES/VARIABLE_VALUEB'variables/42/.ATTRIBUTES/VARIABLE_VALUEB'variables/43/.ATTRIBUTES/VARIABLE_VALUEB'variables/44/.ATTRIBUTES/VARIABLE_VALUEB'variables/45/.ATTRIBUTES/VARIABLE_VALUEB'variables/46/.ATTRIBUTES/VARIABLE_VALUEB'variables/47/.ATTRIBUTES/VARIABLE_VALUEB'variables/48/.ATTRIBUTES/VARIABLE_VALUEB'variables/49/.ATTRIBUTES/VARIABLE_VALUEB'variables/50/.ATTRIBUTES/VARIABLE_VALUEB'variables/51/.ATTRIBUTES/VARIABLE_VALUEB'variables/52/.ATTRIBUTES/VARIABLE_VALUEB'variables/53/.ATTRIBUTES/VARIABLE_VALUEB'variables/54/.ATTRIBUTES/VARIABLE_VALUEB'variables/55/.ATTRIBUTES/VARIABLE_VALUEB'variables/56/.ATTRIBUTES/VARIABLE_VALUEB'variables/57/.ATTRIBUTES/VARIABLE_VALUEB'variables/58/.ATTRIBUTES/VARIABLE_VALUEB'variables/59/.ATTRIBUTES/VARIABLE_VALUEB'variables/60/.ATTRIBUTES/VARIABLE_VALUEB'variables/61/.ATTRIBUTES/VARIABLE_VALUEB'variables/62/.ATTRIBUTES/VARIABLE_VALUEB'variables/63/.ATTRIBUTES/VARIABLE_VALUEB'variables/64/.ATTRIBUTES/VARIABLE_VALUEB'variables/65/.ATTRIBUTES/VARIABLE_VALUEB'variables/66/.ATTRIBUTES/VARIABLE_VALUEB'variables/67/.ATTRIBUTES/VARIABLE_VALUEB'variables/68/.ATTRIBUTES/VARIABLE_VALUEB'variables/69/.ATTRIBUTES/VARIABLE_VALUEB'variables/70/.ATTRIBUTES/VARIABLE_VALUEB'variables/71/.ATTRIBUTES/VARIABLE_VALUEB'variables/72/.ATTRIBUTES/VARIABLE_VALUEB'variables/73/.ATTRIBUTES/VARIABLE_VALUEB'variables/74/.ATTRIBUTES/VARIABLE_VALUEB'variables/75/.ATTRIBUTES/VARIABLE_VALUEB'variables/76/.ATTRIBUTES/VARIABLE_VALUEB'variables/77/.ATTRIBUTES/VARIABLE_VALUEB'variables/78/.ATTRIBUTES/VARIABLE_VALUEB'variables/79/.ATTRIBUTES/VARIABLE_VALUEB'variables/80/.ATTRIBUTES/VARIABLE_VALUEB'variables/81/.ATTRIBUTES/VARIABLE_VALUEB'variables/82/.ATTRIBUTES/VARIABLE_VALUEB'variables/83/.ATTRIBUTES/VARIABLE_VALUEB'variables/84/.ATTRIBUTES/VARIABLE_VALUEB'variables/85/.ATTRIBUTES/VARIABLE_VALUEB'variables/86/.ATTRIBUTES/VARIABLE_VALUEB'variables/87/.ATTRIBUTES/VARIABLE_VALUEB'variables/88/.ATTRIBUTES/VARIABLE_VALUEB'variables/89/.ATTRIBUTES/VARIABLE_VALUEB'variables/90/.ATTRIBUTES/VARIABLE_VALUEB'variables/91/.ATTRIBUTES/VARIABLE_VALUEB'variables/92/.ATTRIBUTES/VARIABLE_VALUEB'variables/93/.ATTRIBUTES/VARIABLE_VALUEB'variables/94/.ATTRIBUTES/VARIABLE_VALUEB'variables/95/.ATTRIBUTES/VARIABLE_VALUEB'variables/96/.ATTRIBUTES/VARIABLE_VALUEB'variables/97/.ATTRIBUTES/VARIABLE_VALUEB'variables/98/.ATTRIBUTES/VARIABLE_VALUEB'variables/99/.ATTRIBUTES/VARIABLE_VALUEB(variables/100/.ATTRIBUTES/VARIABLE_VALUEB(variables/101/.ATTRIBUTES/VARIABLE_VALUEB(variables/102/.ATTRIBUTES/VARIABLE_VALUEB(variables/103/.ATTRIBUTES/VARIABLE_VALUEB(variables/104/.ATTRIBUTES/VARIABLE_VALUEB(variables/105/.ATTRIBUTES/VARIABLE_VALUEB(variables/106/.ATTRIBUTES/VARIABLE_VALUEB(variables/107/.ATTRIBUTES/VARIABLE_VALUEB(variables/108/.ATTRIBUTES/VARIABLE_VALUEB(variables/109/.ATTRIBUTES/VARIABLE_VALUEB(variables/110/.ATTRIBUTES/VARIABLE_VALUEB(variables/111/.ATTRIBUTES/VARIABLE_VALUEB(variables/112/.ATTRIBUTES/VARIABLE_VALUEB(variables/113/.ATTRIBUTES/VARIABLE_VALUEB(variables/114/.ATTRIBUTES/VARIABLE_VALUEB(variables/115/.ATTRIBUTES/VARIABLE_VALUEB(variables/116/.ATTRIBUTES/VARIABLE_VALUEB(variables/117/.ATTRIBUTES/VARIABLE_VALUEB(variables/118/.ATTRIBUTES/VARIABLE_VALUEB(variables/119/.ATTRIBUTES/VARIABLE_VALUEB(variables/120/.ATTRIBUTES/VARIABLE_VALUEB(variables/121/.ATTRIBUTES/VARIABLE_VALUEB(variables/122/.ATTRIBUTES/VARIABLE_VALUEB(variables/123/.ATTRIBUTES/VARIABLE_VALUEB(variables/124/.ATTRIBUTES/VARIABLE_VALUEB(variables/125/.ATTRIBUTES/VARIABLE_VALUEB(variables/126/.ATTRIBUTES/VARIABLE_VALUEB(variables/127/.ATTRIBUTES/VARIABLE_VALUEB(variables/128/.ATTRIBUTES/VARIABLE_VALUEB(variables/129/.ATTRIBUTES/VARIABLE_VALUEB(variables/130/.ATTRIBUTES/VARIABLE_VALUEB(variables/131/.ATTRIBUTES/VARIABLE_VALUEB(variables/132/.ATTRIBUTES/VARIABLE_VALUEB(variables/133/.ATTRIBUTES/VARIABLE_VALUEB(variables/134/.ATTRIBUTES/VARIABLE_VALUEB(variables/135/.ATTRIBUTES/VARIABLE_VALUEB(variables/136/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes	
:?*
dtype0*?
value?B??B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*?
dtypes?
?2?	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0	*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_save_counterIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpKassignvariableop_1_mobilenetv1_conv2d_1_depthwise_batchnorm_moving_varianceIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOpAassignvariableop_2_mobilenetv1_conv2d_3_depthwise_batchnorm_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpAassignvariableop_3_mobilenetv1_conv2d_7_pointwise_batchnorm_gammaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp@assignvariableop_4_mobilenetv1_conv2d_8_pointwise_batchnorm_betaIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpAassignvariableop_5_mobilenetv1_conv2d_11_pointwise_batchnorm_betaIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOpCassignvariableop_6_mobilenetv1_conv2d_4_depthwise_depthwise_weightsIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp;assignvariableop_7_mobilenetv1_logits_conv2d_1c_1x1_weightsIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp=assignvariableop_8_mobilenetv1_conv2d_0_batchnorm_moving_meanIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpGassignvariableop_9_mobilenetv1_conv2d_3_pointwise_batchnorm_moving_meanIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOpHassignvariableop_10_mobilenetv1_conv2d_4_depthwise_batchnorm_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOpLassignvariableop_11_mobilenetv1_conv2d_5_depthwise_batchnorm_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOpBassignvariableop_12_mobilenetv1_conv2d_6_depthwise_batchnorm_gammaIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpLassignvariableop_13_mobilenetv1_conv2d_7_pointwise_batchnorm_moving_varianceIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOpLassignvariableop_14_mobilenetv1_conv2d_9_depthwise_batchnorm_moving_varianceIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpLassignvariableop_15_mobilenetv1_conv2d_9_pointwise_batchnorm_moving_varianceIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp;assignvariableop_16_mobilenetv1_conv2d_11_pointwise_weightsIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOpHassignvariableop_17_mobilenetv1_conv2d_1_pointwise_batchnorm_moving_meanIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOpHassignvariableop_18_mobilenetv1_conv2d_5_depthwise_batchnorm_moving_meanIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOpIassignvariableop_19_mobilenetv1_conv2d_10_pointwise_batchnorm_moving_meanIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOpAassignvariableop_20_mobilenetv1_conv2d_7_pointwise_batchnorm_betaIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp0assignvariableop_21_mobilenetv1_conv2d_0_weightsIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOpLassignvariableop_22_mobilenetv1_conv2d_2_depthwise_batchnorm_moving_varianceIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOpLassignvariableop_23_mobilenetv1_conv2d_4_depthwise_batchnorm_moving_varianceIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOpHassignvariableop_24_mobilenetv1_conv2d_6_pointwise_batchnorm_moving_meanIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOpAassignvariableop_25_mobilenetv1_conv2d_9_depthwise_batchnorm_betaIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOpCassignvariableop_26_mobilenetv1_conv2d_11_depthwise_batchnorm_gammaIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp:assignvariableop_27_mobilenetv1_conv2d_6_pointwise_weightsIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOpAassignvariableop_28_mobilenetv1_conv2d_6_pointwise_batchnorm_betaIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOpAassignvariableop_29_mobilenetv1_conv2d_1_depthwise_batchnorm_betaIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOpBassignvariableop_30_mobilenetv1_conv2d_3_pointwise_batchnorm_gammaIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOpBassignvariableop_31_mobilenetv1_conv2d_10_depthwise_batchnorm_betaIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp;assignvariableop_32_mobilenetv1_logits_conv2d_1c_1x1_biasesIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOpHassignvariableop_33_mobilenetv1_conv2d_8_pointwise_batchnorm_moving_meanIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOpAassignvariableop_34_mobilenetv1_conv2d_5_pointwise_batchnorm_betaIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOpDassignvariableop_35_mobilenetv1_conv2d_2_depthwise_depthwise_weightsIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp:assignvariableop_36_mobilenetv1_conv2d_1_pointwise_weightsIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOpBassignvariableop_37_mobilenetv1_conv2d_1_pointwise_batchnorm_gammaIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOpAassignvariableop_38_mobilenetv1_conv2d_5_depthwise_batchnorm_betaIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOpLassignvariableop_39_mobilenetv1_conv2d_5_pointwise_batchnorm_moving_varianceIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOpBassignvariableop_40_mobilenetv1_conv2d_8_depthwise_batchnorm_gammaIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOpEassignvariableop_41_mobilenetv1_conv2d_13_depthwise_depthwise_weightsIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOpLassignvariableop_42_mobilenetv1_conv2d_6_depthwise_batchnorm_moving_varianceIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp:assignvariableop_43_mobilenetv1_conv2d_9_pointwise_weightsIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOpMassignvariableop_44_mobilenetv1_conv2d_10_depthwise_batchnorm_moving_varianceIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOpBassignvariableop_45_mobilenetv1_conv2d_12_pointwise_batchnorm_betaIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOpAassignvariableop_46_mobilenetv1_conv2d_2_depthwise_batchnorm_betaIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOpHassignvariableop_47_mobilenetv1_conv2d_2_pointwise_batchnorm_moving_meanIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOpAassignvariableop_48_mobilenetv1_conv2d_3_pointwise_batchnorm_betaIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOpHassignvariableop_49_mobilenetv1_conv2d_6_depthwise_batchnorm_moving_meanIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOpCassignvariableop_50_mobilenetv1_conv2d_10_pointwise_batchnorm_gammaIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51?
AssignVariableOp_51AssignVariableOpCassignvariableop_51_mobilenetv1_conv2d_12_depthwise_batchnorm_gammaIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52?
AssignVariableOp_52AssignVariableOpIassignvariableop_52_mobilenetv1_conv2d_13_depthwise_batchnorm_moving_meanIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53?
AssignVariableOp_53AssignVariableOpDassignvariableop_53_mobilenetv1_conv2d_5_depthwise_depthwise_weightsIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54?
AssignVariableOp_54AssignVariableOpBassignvariableop_54_mobilenetv1_conv2d_13_depthwise_batchnorm_betaIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55?
AssignVariableOp_55AssignVariableOpBassignvariableop_55_mobilenetv1_conv2d_8_pointwise_batchnorm_gammaIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56?
AssignVariableOp_56AssignVariableOpAassignvariableop_56_mobilenetv1_conv2d_3_depthwise_batchnorm_betaIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57?
AssignVariableOp_57AssignVariableOpBassignvariableop_57_mobilenetv1_conv2d_7_depthwise_batchnorm_gammaIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58?
AssignVariableOp_58AssignVariableOpLassignvariableop_58_mobilenetv1_conv2d_8_depthwise_batchnorm_moving_varianceIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59?
AssignVariableOp_59AssignVariableOpIassignvariableop_59_mobilenetv1_conv2d_10_depthwise_batchnorm_moving_meanIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60?
AssignVariableOp_60AssignVariableOpMassignvariableop_60_mobilenetv1_conv2d_10_pointwise_batchnorm_moving_varianceIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61?
AssignVariableOp_61AssignVariableOp;assignvariableop_61_mobilenetv1_conv2d_12_pointwise_weightsIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62?
AssignVariableOp_62AssignVariableOpMassignvariableop_62_mobilenetv1_conv2d_12_pointwise_batchnorm_moving_varianceIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63?
AssignVariableOp_63AssignVariableOpBassignvariableop_63_mobilenetv1_conv2d_4_depthwise_batchnorm_gammaIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64?
AssignVariableOp_64AssignVariableOpHassignvariableop_64_mobilenetv1_conv2d_8_depthwise_batchnorm_moving_meanIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65?
AssignVariableOp_65AssignVariableOpIassignvariableop_65_mobilenetv1_conv2d_13_pointwise_batchnorm_moving_meanIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66?
AssignVariableOp_66AssignVariableOpLassignvariableop_66_mobilenetv1_conv2d_7_depthwise_batchnorm_moving_varianceIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67?
AssignVariableOp_67AssignVariableOp:assignvariableop_67_mobilenetv1_conv2d_7_pointwise_weightsIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68?
AssignVariableOp_68AssignVariableOp:assignvariableop_68_mobilenetv1_conv2d_5_pointwise_weightsIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69?
AssignVariableOp_69AssignVariableOpLassignvariableop_69_mobilenetv1_conv2d_1_pointwise_batchnorm_moving_varianceIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70?
AssignVariableOp_70AssignVariableOpHassignvariableop_70_mobilenetv1_conv2d_1_depthwise_batchnorm_moving_meanIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71?
AssignVariableOp_71AssignVariableOpLassignvariableop_71_mobilenetv1_conv2d_3_pointwise_batchnorm_moving_varianceIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72?
AssignVariableOp_72AssignVariableOpHassignvariableop_72_mobilenetv1_conv2d_9_pointwise_batchnorm_moving_meanIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73?
AssignVariableOp_73AssignVariableOpBassignvariableop_73_mobilenetv1_conv2d_10_pointwise_batchnorm_betaIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_73n
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:2
Identity_74?
AssignVariableOp_74AssignVariableOpMassignvariableop_74_mobilenetv1_conv2d_12_depthwise_batchnorm_moving_varianceIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_74n
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:2
Identity_75?
AssignVariableOp_75AssignVariableOpHassignvariableop_75_mobilenetv1_conv2d_4_pointwise_batchnorm_moving_meanIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_75n
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:2
Identity_76?
AssignVariableOp_76AssignVariableOp;assignvariableop_76_mobilenetv1_conv2d_10_pointwise_weightsIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_76n
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:2
Identity_77?
AssignVariableOp_77AssignVariableOpDassignvariableop_77_mobilenetv1_conv2d_1_depthwise_depthwise_weightsIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_77n
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:2
Identity_78?
AssignVariableOp_78AssignVariableOpAassignvariableop_78_mobilenetv1_conv2d_1_pointwise_batchnorm_betaIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_78n
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:2
Identity_79?
AssignVariableOp_79AssignVariableOpBassignvariableop_79_mobilenetv1_conv2d_0_batchnorm_moving_varianceIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79n
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:2
Identity_80?
AssignVariableOp_80AssignVariableOpBassignvariableop_80_mobilenetv1_conv2d_2_pointwise_batchnorm_gammaIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_80n
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:2
Identity_81?
AssignVariableOp_81AssignVariableOpDassignvariableop_81_mobilenetv1_conv2d_9_depthwise_depthwise_weightsIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_81n
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:2
Identity_82?
AssignVariableOp_82AssignVariableOpIassignvariableop_82_mobilenetv1_conv2d_11_pointwise_batchnorm_moving_meanIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_82n
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:2
Identity_83?
AssignVariableOp_83AssignVariableOpDassignvariableop_83_mobilenetv1_conv2d_8_depthwise_depthwise_weightsIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_83n
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:2
Identity_84?
AssignVariableOp_84AssignVariableOpAassignvariableop_84_mobilenetv1_conv2d_8_depthwise_batchnorm_betaIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_84n
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:2
Identity_85?
AssignVariableOp_85AssignVariableOpLassignvariableop_85_mobilenetv1_conv2d_8_pointwise_batchnorm_moving_varianceIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_85n
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:2
Identity_86?
AssignVariableOp_86AssignVariableOp:assignvariableop_86_mobilenetv1_conv2d_3_pointwise_weightsIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_86n
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:2
Identity_87?
AssignVariableOp_87AssignVariableOpBassignvariableop_87_mobilenetv1_conv2d_12_depthwise_batchnorm_betaIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_87n
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:2
Identity_88?
AssignVariableOp_88AssignVariableOpMassignvariableop_88_mobilenetv1_conv2d_13_depthwise_batchnorm_moving_varianceIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_88n
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:2
Identity_89?
AssignVariableOp_89AssignVariableOp;assignvariableop_89_mobilenetv1_conv2d_13_pointwise_weightsIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_89n
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:2
Identity_90?
AssignVariableOp_90AssignVariableOpDassignvariableop_90_mobilenetv1_conv2d_7_depthwise_depthwise_weightsIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_90n
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:2
Identity_91?
AssignVariableOp_91AssignVariableOpAassignvariableop_91_mobilenetv1_conv2d_7_depthwise_batchnorm_betaIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_91n
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:2
Identity_92?
AssignVariableOp_92AssignVariableOpCassignvariableop_92_mobilenetv1_conv2d_13_pointwise_batchnorm_gammaIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_92n
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:2
Identity_93?
AssignVariableOp_93AssignVariableOp8assignvariableop_93_mobilenetv1_conv2d_0_batchnorm_gammaIdentity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_93n
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:2
Identity_94?
AssignVariableOp_94AssignVariableOpAassignvariableop_94_mobilenetv1_conv2d_4_depthwise_batchnorm_betaIdentity_94:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_94n
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:2
Identity_95?
AssignVariableOp_95AssignVariableOpAassignvariableop_95_mobilenetv1_conv2d_9_pointwise_batchnorm_betaIdentity_95:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_95n
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:2
Identity_96?
AssignVariableOp_96AssignVariableOpBassignvariableop_96_mobilenetv1_conv2d_5_pointwise_batchnorm_gammaIdentity_96:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_96n
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:2
Identity_97?
AssignVariableOp_97AssignVariableOpBassignvariableop_97_mobilenetv1_conv2d_2_depthwise_batchnorm_gammaIdentity_97:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_97n
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:2
Identity_98?
AssignVariableOp_98AssignVariableOpBassignvariableop_98_mobilenetv1_conv2d_4_pointwise_batchnorm_gammaIdentity_98:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_98n
Identity_99IdentityRestoreV2:tensors:99"/device:CPU:0*
T0*
_output_shapes
:2
Identity_99?
AssignVariableOp_99AssignVariableOpDassignvariableop_99_mobilenetv1_conv2d_6_depthwise_depthwise_weightsIdentity_99:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_99q
Identity_100IdentityRestoreV2:tensors:100"/device:CPU:0*
T0*
_output_shapes
:2
Identity_100?
AssignVariableOp_100AssignVariableOpCassignvariableop_100_mobilenetv1_conv2d_9_depthwise_batchnorm_gammaIdentity_100:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_100q
Identity_101IdentityRestoreV2:tensors:101"/device:CPU:0*
T0*
_output_shapes
:2
Identity_101?
AssignVariableOp_101AssignVariableOpNassignvariableop_101_mobilenetv1_conv2d_11_depthwise_batchnorm_moving_varianceIdentity_101:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_101q
Identity_102IdentityRestoreV2:tensors:102"/device:CPU:0*
T0*
_output_shapes
:2
Identity_102?
AssignVariableOp_102AssignVariableOpDassignvariableop_102_mobilenetv1_conv2d_13_depthwise_batchnorm_gammaIdentity_102:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_102q
Identity_103IdentityRestoreV2:tensors:103"/device:CPU:0*
T0*
_output_shapes
:2
Identity_103?
AssignVariableOp_103AssignVariableOpNassignvariableop_103_mobilenetv1_conv2d_13_pointwise_batchnorm_moving_varianceIdentity_103:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_103q
Identity_104IdentityRestoreV2:tensors:104"/device:CPU:0*
T0*
_output_shapes
:2
Identity_104?
AssignVariableOp_104AssignVariableOpJassignvariableop_104_mobilenetv1_conv2d_11_depthwise_batchnorm_moving_meanIdentity_104:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_104q
Identity_105IdentityRestoreV2:tensors:105"/device:CPU:0*
T0*
_output_shapes
:2
Identity_105?
AssignVariableOp_105AssignVariableOp;assignvariableop_105_mobilenetv1_conv2d_8_pointwise_weightsIdentity_105:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_105q
Identity_106IdentityRestoreV2:tensors:106"/device:CPU:0*
T0*
_output_shapes
:2
Identity_106?
AssignVariableOp_106AssignVariableOpIassignvariableop_106_mobilenetv1_conv2d_3_depthwise_batchnorm_moving_meanIdentity_106:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_106q
Identity_107IdentityRestoreV2:tensors:107"/device:CPU:0*
T0*
_output_shapes
:2
Identity_107?
AssignVariableOp_107AssignVariableOp;assignvariableop_107_mobilenetv1_conv2d_4_pointwise_weightsIdentity_107:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_107q
Identity_108IdentityRestoreV2:tensors:108"/device:CPU:0*
T0*
_output_shapes
:2
Identity_108?
AssignVariableOp_108AssignVariableOpMassignvariableop_108_mobilenetv1_conv2d_4_pointwise_batchnorm_moving_varianceIdentity_108:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_108q
Identity_109IdentityRestoreV2:tensors:109"/device:CPU:0*
T0*
_output_shapes
:2
Identity_109?
AssignVariableOp_109AssignVariableOpBassignvariableop_109_mobilenetv1_conv2d_6_depthwise_batchnorm_betaIdentity_109:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_109q
Identity_110IdentityRestoreV2:tensors:110"/device:CPU:0*
T0*
_output_shapes
:2
Identity_110?
AssignVariableOp_110AssignVariableOpMassignvariableop_110_mobilenetv1_conv2d_6_pointwise_batchnorm_moving_varianceIdentity_110:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_110q
Identity_111IdentityRestoreV2:tensors:111"/device:CPU:0*
T0*
_output_shapes
:2
Identity_111?
AssignVariableOp_111AssignVariableOpIassignvariableop_111_mobilenetv1_conv2d_9_depthwise_batchnorm_moving_meanIdentity_111:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_111q
Identity_112IdentityRestoreV2:tensors:112"/device:CPU:0*
T0*
_output_shapes
:2
Identity_112?
AssignVariableOp_112AssignVariableOpCassignvariableop_112_mobilenetv1_conv2d_9_pointwise_batchnorm_gammaIdentity_112:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_112q
Identity_113IdentityRestoreV2:tensors:113"/device:CPU:0*
T0*
_output_shapes
:2
Identity_113?
AssignVariableOp_113AssignVariableOpJassignvariableop_113_mobilenetv1_conv2d_12_pointwise_batchnorm_moving_meanIdentity_113:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_113q
Identity_114IdentityRestoreV2:tensors:114"/device:CPU:0*
T0*
_output_shapes
:2
Identity_114?
AssignVariableOp_114AssignVariableOpCassignvariableop_114_mobilenetv1_conv2d_13_pointwise_batchnorm_betaIdentity_114:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_114q
Identity_115IdentityRestoreV2:tensors:115"/device:CPU:0*
T0*
_output_shapes
:2
Identity_115?
AssignVariableOp_115AssignVariableOpIassignvariableop_115_mobilenetv1_conv2d_7_pointwise_batchnorm_moving_meanIdentity_115:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_115q
Identity_116IdentityRestoreV2:tensors:116"/device:CPU:0*
T0*
_output_shapes
:2
Identity_116?
AssignVariableOp_116AssignVariableOpJassignvariableop_116_mobilenetv1_conv2d_12_depthwise_batchnorm_moving_meanIdentity_116:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_116q
Identity_117IdentityRestoreV2:tensors:117"/device:CPU:0*
T0*
_output_shapes
:2
Identity_117?
AssignVariableOp_117AssignVariableOpEassignvariableop_117_mobilenetv1_conv2d_3_depthwise_depthwise_weightsIdentity_117:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_117q
Identity_118IdentityRestoreV2:tensors:118"/device:CPU:0*
T0*
_output_shapes
:2
Identity_118?
AssignVariableOp_118AssignVariableOpBassignvariableop_118_mobilenetv1_conv2d_4_pointwise_batchnorm_betaIdentity_118:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_118q
Identity_119IdentityRestoreV2:tensors:119"/device:CPU:0*
T0*
_output_shapes
:2
Identity_119?
AssignVariableOp_119AssignVariableOpIassignvariableop_119_mobilenetv1_conv2d_2_depthwise_batchnorm_moving_meanIdentity_119:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_119q
Identity_120IdentityRestoreV2:tensors:120"/device:CPU:0*
T0*
_output_shapes
:2
Identity_120?
AssignVariableOp_120AssignVariableOpBassignvariableop_120_mobilenetv1_conv2d_2_pointwise_batchnorm_betaIdentity_120:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_120q
Identity_121IdentityRestoreV2:tensors:121"/device:CPU:0*
T0*
_output_shapes
:2
Identity_121?
AssignVariableOp_121AssignVariableOpDassignvariableop_121_mobilenetv1_conv2d_10_depthwise_batchnorm_gammaIdentity_121:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_121q
Identity_122IdentityRestoreV2:tensors:122"/device:CPU:0*
T0*
_output_shapes
:2
Identity_122?
AssignVariableOp_122AssignVariableOpFassignvariableop_122_mobilenetv1_conv2d_12_depthwise_depthwise_weightsIdentity_122:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_122q
Identity_123IdentityRestoreV2:tensors:123"/device:CPU:0*
T0*
_output_shapes
:2
Identity_123?
AssignVariableOp_123AssignVariableOp8assignvariableop_123_mobilenetv1_conv2d_0_batchnorm_betaIdentity_123:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_123q
Identity_124IdentityRestoreV2:tensors:124"/device:CPU:0*
T0*
_output_shapes
:2
Identity_124?
AssignVariableOp_124AssignVariableOpFassignvariableop_124_mobilenetv1_conv2d_11_depthwise_depthwise_weightsIdentity_124:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_124q
Identity_125IdentityRestoreV2:tensors:125"/device:CPU:0*
T0*
_output_shapes
:2
Identity_125?
AssignVariableOp_125AssignVariableOpCassignvariableop_125_mobilenetv1_conv2d_11_depthwise_batchnorm_betaIdentity_125:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_125q
Identity_126IdentityRestoreV2:tensors:126"/device:CPU:0*
T0*
_output_shapes
:2
Identity_126?
AssignVariableOp_126AssignVariableOpNassignvariableop_126_mobilenetv1_conv2d_11_pointwise_batchnorm_moving_varianceIdentity_126:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_126q
Identity_127IdentityRestoreV2:tensors:127"/device:CPU:0*
T0*
_output_shapes
:2
Identity_127?
AssignVariableOp_127AssignVariableOpIassignvariableop_127_mobilenetv1_conv2d_7_depthwise_batchnorm_moving_meanIdentity_127:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_127q
Identity_128IdentityRestoreV2:tensors:128"/device:CPU:0*
T0*
_output_shapes
:2
Identity_128?
AssignVariableOp_128AssignVariableOpCassignvariableop_128_mobilenetv1_conv2d_1_depthwise_batchnorm_gammaIdentity_128:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_128q
Identity_129IdentityRestoreV2:tensors:129"/device:CPU:0*
T0*
_output_shapes
:2
Identity_129?
AssignVariableOp_129AssignVariableOpMassignvariableop_129_mobilenetv1_conv2d_3_depthwise_batchnorm_moving_varianceIdentity_129:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_129q
Identity_130IdentityRestoreV2:tensors:130"/device:CPU:0*
T0*
_output_shapes
:2
Identity_130?
AssignVariableOp_130AssignVariableOpIassignvariableop_130_mobilenetv1_conv2d_5_pointwise_batchnorm_moving_meanIdentity_130:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_130q
Identity_131IdentityRestoreV2:tensors:131"/device:CPU:0*
T0*
_output_shapes
:2
Identity_131?
AssignVariableOp_131AssignVariableOpDassignvariableop_131_mobilenetv1_conv2d_12_pointwise_batchnorm_gammaIdentity_131:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_131q
Identity_132IdentityRestoreV2:tensors:132"/device:CPU:0*
T0*
_output_shapes
:2
Identity_132?
AssignVariableOp_132AssignVariableOp;assignvariableop_132_mobilenetv1_conv2d_2_pointwise_weightsIdentity_132:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_132q
Identity_133IdentityRestoreV2:tensors:133"/device:CPU:0*
T0*
_output_shapes
:2
Identity_133?
AssignVariableOp_133AssignVariableOpMassignvariableop_133_mobilenetv1_conv2d_2_pointwise_batchnorm_moving_varianceIdentity_133:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_133q
Identity_134IdentityRestoreV2:tensors:134"/device:CPU:0*
T0*
_output_shapes
:2
Identity_134?
AssignVariableOp_134AssignVariableOpCassignvariableop_134_mobilenetv1_conv2d_5_depthwise_batchnorm_gammaIdentity_134:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_134q
Identity_135IdentityRestoreV2:tensors:135"/device:CPU:0*
T0*
_output_shapes
:2
Identity_135?
AssignVariableOp_135AssignVariableOpFassignvariableop_135_mobilenetv1_conv2d_10_depthwise_depthwise_weightsIdentity_135:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_135q
Identity_136IdentityRestoreV2:tensors:136"/device:CPU:0*
T0*
_output_shapes
:2
Identity_136?
AssignVariableOp_136AssignVariableOpDassignvariableop_136_mobilenetv1_conv2d_11_pointwise_batchnorm_gammaIdentity_136:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_136q
Identity_137IdentityRestoreV2:tensors:137"/device:CPU:0*
T0*
_output_shapes
:2
Identity_137?
AssignVariableOp_137AssignVariableOpCassignvariableop_137_mobilenetv1_conv2d_6_pointwise_batchnorm_gammaIdentity_137:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1379
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_138Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_125^AssignVariableOp_126^AssignVariableOp_127^AssignVariableOp_128^AssignVariableOp_129^AssignVariableOp_13^AssignVariableOp_130^AssignVariableOp_131^AssignVariableOp_132^AssignVariableOp_133^AssignVariableOp_134^AssignVariableOp_135^AssignVariableOp_136^AssignVariableOp_137^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_138i
Identity_139IdentityIdentity_138:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_139?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_125^AssignVariableOp_126^AssignVariableOp_127^AssignVariableOp_128^AssignVariableOp_129^AssignVariableOp_13^AssignVariableOp_130^AssignVariableOp_131^AssignVariableOp_132^AssignVariableOp_133^AssignVariableOp_134^AssignVariableOp_135^AssignVariableOp_136^AssignVariableOp_137^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99*
_output_shapes
 2
NoOp_1"%
identity_139Identity_139:output:0*?
_input_shapes?
?: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102,
AssignVariableOp_100AssignVariableOp_1002,
AssignVariableOp_101AssignVariableOp_1012,
AssignVariableOp_102AssignVariableOp_1022,
AssignVariableOp_103AssignVariableOp_1032,
AssignVariableOp_104AssignVariableOp_1042,
AssignVariableOp_105AssignVariableOp_1052,
AssignVariableOp_106AssignVariableOp_1062,
AssignVariableOp_107AssignVariableOp_1072,
AssignVariableOp_108AssignVariableOp_1082,
AssignVariableOp_109AssignVariableOp_1092*
AssignVariableOp_11AssignVariableOp_112,
AssignVariableOp_110AssignVariableOp_1102,
AssignVariableOp_111AssignVariableOp_1112,
AssignVariableOp_112AssignVariableOp_1122,
AssignVariableOp_113AssignVariableOp_1132,
AssignVariableOp_114AssignVariableOp_1142,
AssignVariableOp_115AssignVariableOp_1152,
AssignVariableOp_116AssignVariableOp_1162,
AssignVariableOp_117AssignVariableOp_1172,
AssignVariableOp_118AssignVariableOp_1182,
AssignVariableOp_119AssignVariableOp_1192*
AssignVariableOp_12AssignVariableOp_122,
AssignVariableOp_120AssignVariableOp_1202,
AssignVariableOp_121AssignVariableOp_1212,
AssignVariableOp_122AssignVariableOp_1222,
AssignVariableOp_123AssignVariableOp_1232,
AssignVariableOp_124AssignVariableOp_1242,
AssignVariableOp_125AssignVariableOp_1252,
AssignVariableOp_126AssignVariableOp_1262,
AssignVariableOp_127AssignVariableOp_1272,
AssignVariableOp_128AssignVariableOp_1282,
AssignVariableOp_129AssignVariableOp_1292*
AssignVariableOp_13AssignVariableOp_132,
AssignVariableOp_130AssignVariableOp_1302,
AssignVariableOp_131AssignVariableOp_1312,
AssignVariableOp_132AssignVariableOp_1322,
AssignVariableOp_133AssignVariableOp_1332,
AssignVariableOp_134AssignVariableOp_1342,
AssignVariableOp_135AssignVariableOp_1352,
AssignVariableOp_136AssignVariableOp_1362,
AssignVariableOp_137AssignVariableOp_1372*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862*
AssignVariableOp_87AssignVariableOp_872*
AssignVariableOp_88AssignVariableOp_882*
AssignVariableOp_89AssignVariableOp_892(
AssignVariableOp_9AssignVariableOp_92*
AssignVariableOp_90AssignVariableOp_902*
AssignVariableOp_91AssignVariableOp_912*
AssignVariableOp_92AssignVariableOp_922*
AssignVariableOp_93AssignVariableOp_932*
AssignVariableOp_94AssignVariableOp_942*
AssignVariableOp_95AssignVariableOp_952*
AssignVariableOp_96AssignVariableOp_962*
AssignVariableOp_97AssignVariableOp_972*
AssignVariableOp_98AssignVariableOp_982*
AssignVariableOp_99AssignVariableOp_99:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
S
inputsI
serving_default_inputs:0+???????????????????????????;
logits1
StatefulPartitionedCall:0??????????tensorflow/serving/predict:??
?
	variables
trainable_variables
regularization_losses
save_counter

signatures
?__call__"
_generic_user_object
?	
0
1
2
	3

4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
 26
!27
"28
#29
$30
%31
&32
'33
(34
)35
*36
+37
,38
-39
.40
/41
042
143
244
345
446
547
648
749
850
951
:52
;53
<54
=55
>56
?57
@58
A59
B60
C61
D62
E63
F64
G65
H66
I67
J68
K69
L70
M71
N72
O73
P74
Q75
R76
S77
T78
U79
V80
W81
X82
Y83
Z84
[85
\86
]87
^88
_89
`90
a91
b92
c93
d94
e95
f96
g97
h98
i99
j100
k101
l102
m103
n104
o105
p106
q107
r108
s109
t110
u111
v112
w113
x114
y115
z116
{117
|118
}119
~120
121
?122
?123
?124
?125
?126
?127
?128
?129
?130
?131
?132
?133
?134
?135
?136"
trackable_list_wrapper
?
0
1
	2

3
4
5
6
7
8
9
10
11
 12
!13
"14
#15
$16
%17
'18
(19
)20
*21
+22
-23
.24
025
226
327
528
729
830
:31
;32
<33
=34
>35
B36
D37
H38
I39
N40
Q41
R42
S43
U44
V45
X46
Y47
[48
\49
^50
_51
`52
a53
b54
c55
d56
e57
f58
g59
h60
i61
k62
n63
p64
r65
u66
w67
z68
{69
}70
~71
72
?73
?74
?75
?76
?77
?78
?79
?80
?81
?82"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
:	 2save_counter
-
?serving_default"
signature_map
H:F (28MobilenetV1/Conv2d_1_depthwise/BatchNorm/moving_variance
<::`2.MobilenetV1/Conv2d_3_depthwise/BatchNorm/gamma
=:;?2.MobilenetV1/Conv2d_7_pointwise/BatchNorm/gamma
<::?2-MobilenetV1/Conv2d_8_pointwise/BatchNorm/beta
=:;?2.MobilenetV1/Conv2d_11_pointwise/BatchNorm/beta
J:H`20MobilenetV1/Conv2d_4_depthwise/depthwise_weights
D:B??2(MobilenetV1/Logits/Conv2d_1c_1x1/weights
::8 (2*MobilenetV1/Conv2d_0/BatchNorm/moving_mean
D:B` (24MobilenetV1/Conv2d_3_pointwise/BatchNorm/moving_mean
D:B` (24MobilenetV1/Conv2d_4_depthwise/BatchNorm/moving_mean
I:G? (28MobilenetV1/Conv2d_5_depthwise/BatchNorm/moving_variance
=:;?2.MobilenetV1/Conv2d_6_depthwise/BatchNorm/gamma
I:G? (28MobilenetV1/Conv2d_7_pointwise/BatchNorm/moving_variance
I:G? (28MobilenetV1/Conv2d_9_depthwise/BatchNorm/moving_variance
I:G? (28MobilenetV1/Conv2d_9_pointwise/BatchNorm/moving_variance
C:A??2'MobilenetV1/Conv2d_11_pointwise/weights
D:B0 (24MobilenetV1/Conv2d_1_pointwise/BatchNorm/moving_mean
E:C? (24MobilenetV1/Conv2d_5_depthwise/BatchNorm/moving_mean
F:D? (25MobilenetV1/Conv2d_10_pointwise/BatchNorm/moving_mean
<::?2-MobilenetV1/Conv2d_7_pointwise/BatchNorm/beta
6:42MobilenetV1/Conv2d_0/weights
H:F0 (28MobilenetV1/Conv2d_2_depthwise/BatchNorm/moving_variance
H:F` (28MobilenetV1/Conv2d_4_depthwise/BatchNorm/moving_variance
E:C? (24MobilenetV1/Conv2d_6_pointwise/BatchNorm/moving_mean
<::?2-MobilenetV1/Conv2d_9_depthwise/BatchNorm/beta
>:<?2/MobilenetV1/Conv2d_11_depthwise/BatchNorm/gamma
B:@??2&MobilenetV1/Conv2d_6_pointwise/weights
<::?2-MobilenetV1/Conv2d_6_pointwise/BatchNorm/beta
;:92-MobilenetV1/Conv2d_1_depthwise/BatchNorm/beta
<::`2.MobilenetV1/Conv2d_3_pointwise/BatchNorm/gamma
=:;?2.MobilenetV1/Conv2d_10_depthwise/BatchNorm/beta
6:4?2'MobilenetV1/Logits/Conv2d_1c_1x1/biases
E:C? (24MobilenetV1/Conv2d_8_pointwise/BatchNorm/moving_mean
<::?2-MobilenetV1/Conv2d_5_pointwise/BatchNorm/beta
J:H020MobilenetV1/Conv2d_2_depthwise/depthwise_weights
@:>02&MobilenetV1/Conv2d_1_pointwise/weights
<::02.MobilenetV1/Conv2d_1_pointwise/BatchNorm/gamma
<::?2-MobilenetV1/Conv2d_5_depthwise/BatchNorm/beta
I:G? (28MobilenetV1/Conv2d_5_pointwise/BatchNorm/moving_variance
=:;?2.MobilenetV1/Conv2d_8_depthwise/BatchNorm/gamma
L:J?21MobilenetV1/Conv2d_13_depthwise/depthwise_weights
I:G? (28MobilenetV1/Conv2d_6_depthwise/BatchNorm/moving_variance
B:@??2&MobilenetV1/Conv2d_9_pointwise/weights
J:H? (29MobilenetV1/Conv2d_10_depthwise/BatchNorm/moving_variance
=:;?2.MobilenetV1/Conv2d_12_pointwise/BatchNorm/beta
;:902-MobilenetV1/Conv2d_2_depthwise/BatchNorm/beta
D:B` (24MobilenetV1/Conv2d_2_pointwise/BatchNorm/moving_mean
;:9`2-MobilenetV1/Conv2d_3_pointwise/BatchNorm/beta
E:C? (24MobilenetV1/Conv2d_6_depthwise/BatchNorm/moving_mean
>:<?2/MobilenetV1/Conv2d_10_pointwise/BatchNorm/gamma
>:<?2/MobilenetV1/Conv2d_12_depthwise/BatchNorm/gamma
F:D? (25MobilenetV1/Conv2d_13_depthwise/BatchNorm/moving_mean
K:I?20MobilenetV1/Conv2d_5_depthwise/depthwise_weights
=:;?2.MobilenetV1/Conv2d_13_depthwise/BatchNorm/beta
=:;?2.MobilenetV1/Conv2d_8_pointwise/BatchNorm/gamma
;:9`2-MobilenetV1/Conv2d_3_depthwise/BatchNorm/beta
=:;?2.MobilenetV1/Conv2d_7_depthwise/BatchNorm/gamma
I:G? (28MobilenetV1/Conv2d_8_depthwise/BatchNorm/moving_variance
F:D? (25MobilenetV1/Conv2d_10_depthwise/BatchNorm/moving_mean
J:H? (29MobilenetV1/Conv2d_10_pointwise/BatchNorm/moving_variance
C:A??2'MobilenetV1/Conv2d_12_pointwise/weights
J:H? (29MobilenetV1/Conv2d_12_pointwise/BatchNorm/moving_variance
<::`2.MobilenetV1/Conv2d_4_depthwise/BatchNorm/gamma
E:C? (24MobilenetV1/Conv2d_8_depthwise/BatchNorm/moving_mean
F:D? (25MobilenetV1/Conv2d_13_pointwise/BatchNorm/moving_mean
I:G? (28MobilenetV1/Conv2d_7_depthwise/BatchNorm/moving_variance
B:@??2&MobilenetV1/Conv2d_7_pointwise/weights
B:@??2&MobilenetV1/Conv2d_5_pointwise/weights
H:F0 (28MobilenetV1/Conv2d_1_pointwise/BatchNorm/moving_variance
D:B (24MobilenetV1/Conv2d_1_depthwise/BatchNorm/moving_mean
H:F` (28MobilenetV1/Conv2d_3_pointwise/BatchNorm/moving_variance
E:C? (24MobilenetV1/Conv2d_9_pointwise/BatchNorm/moving_mean
=:;?2.MobilenetV1/Conv2d_10_pointwise/BatchNorm/beta
J:H? (29MobilenetV1/Conv2d_12_depthwise/BatchNorm/moving_variance
E:C? (24MobilenetV1/Conv2d_4_pointwise/BatchNorm/moving_mean
C:A??2'MobilenetV1/Conv2d_10_pointwise/weights
J:H20MobilenetV1/Conv2d_1_depthwise/depthwise_weights
;:902-MobilenetV1/Conv2d_1_pointwise/BatchNorm/beta
>:< (2.MobilenetV1/Conv2d_0/BatchNorm/moving_variance
<::`2.MobilenetV1/Conv2d_2_pointwise/BatchNorm/gamma
K:I?20MobilenetV1/Conv2d_9_depthwise/depthwise_weights
F:D? (25MobilenetV1/Conv2d_11_pointwise/BatchNorm/moving_mean
K:I?20MobilenetV1/Conv2d_8_depthwise/depthwise_weights
<::?2-MobilenetV1/Conv2d_8_depthwise/BatchNorm/beta
I:G? (28MobilenetV1/Conv2d_8_pointwise/BatchNorm/moving_variance
@:>``2&MobilenetV1/Conv2d_3_pointwise/weights
=:;?2.MobilenetV1/Conv2d_12_depthwise/BatchNorm/beta
J:H? (29MobilenetV1/Conv2d_13_depthwise/BatchNorm/moving_variance
C:A??2'MobilenetV1/Conv2d_13_pointwise/weights
K:I?20MobilenetV1/Conv2d_7_depthwise/depthwise_weights
<::?2-MobilenetV1/Conv2d_7_depthwise/BatchNorm/beta
>:<?2/MobilenetV1/Conv2d_13_pointwise/BatchNorm/gamma
2:02$MobilenetV1/Conv2d_0/BatchNorm/gamma
;:9`2-MobilenetV1/Conv2d_4_depthwise/BatchNorm/beta
<::?2-MobilenetV1/Conv2d_9_pointwise/BatchNorm/beta
=:;?2.MobilenetV1/Conv2d_5_pointwise/BatchNorm/gamma
<::02.MobilenetV1/Conv2d_2_depthwise/BatchNorm/gamma
=:;?2.MobilenetV1/Conv2d_4_pointwise/BatchNorm/gamma
K:I?20MobilenetV1/Conv2d_6_depthwise/depthwise_weights
=:;?2.MobilenetV1/Conv2d_9_depthwise/BatchNorm/gamma
J:H? (29MobilenetV1/Conv2d_11_depthwise/BatchNorm/moving_variance
>:<?2/MobilenetV1/Conv2d_13_depthwise/BatchNorm/gamma
J:H? (29MobilenetV1/Conv2d_13_pointwise/BatchNorm/moving_variance
F:D? (25MobilenetV1/Conv2d_11_depthwise/BatchNorm/moving_mean
B:@??2&MobilenetV1/Conv2d_8_pointwise/weights
D:B` (24MobilenetV1/Conv2d_3_depthwise/BatchNorm/moving_mean
A:?`?2&MobilenetV1/Conv2d_4_pointwise/weights
I:G? (28MobilenetV1/Conv2d_4_pointwise/BatchNorm/moving_variance
<::?2-MobilenetV1/Conv2d_6_depthwise/BatchNorm/beta
I:G? (28MobilenetV1/Conv2d_6_pointwise/BatchNorm/moving_variance
E:C? (24MobilenetV1/Conv2d_9_depthwise/BatchNorm/moving_mean
=:;?2.MobilenetV1/Conv2d_9_pointwise/BatchNorm/gamma
F:D? (25MobilenetV1/Conv2d_12_pointwise/BatchNorm/moving_mean
=:;?2.MobilenetV1/Conv2d_13_pointwise/BatchNorm/beta
E:C? (24MobilenetV1/Conv2d_7_pointwise/BatchNorm/moving_mean
F:D? (25MobilenetV1/Conv2d_12_depthwise/BatchNorm/moving_mean
J:H`20MobilenetV1/Conv2d_3_depthwise/depthwise_weights
<::?2-MobilenetV1/Conv2d_4_pointwise/BatchNorm/beta
D:B0 (24MobilenetV1/Conv2d_2_depthwise/BatchNorm/moving_mean
;:9`2-MobilenetV1/Conv2d_2_pointwise/BatchNorm/beta
>:<?2/MobilenetV1/Conv2d_10_depthwise/BatchNorm/gamma
L:J?21MobilenetV1/Conv2d_12_depthwise/depthwise_weights
1:/2#MobilenetV1/Conv2d_0/BatchNorm/beta
L:J?21MobilenetV1/Conv2d_11_depthwise/depthwise_weights
=:;?2.MobilenetV1/Conv2d_11_depthwise/BatchNorm/beta
J:H? (29MobilenetV1/Conv2d_11_pointwise/BatchNorm/moving_variance
E:C? (24MobilenetV1/Conv2d_7_depthwise/BatchNorm/moving_mean
<::2.MobilenetV1/Conv2d_1_depthwise/BatchNorm/gamma
H:F` (28MobilenetV1/Conv2d_3_depthwise/BatchNorm/moving_variance
E:C? (24MobilenetV1/Conv2d_5_pointwise/BatchNorm/moving_mean
>:<?2/MobilenetV1/Conv2d_12_pointwise/BatchNorm/gamma
@:>0`2&MobilenetV1/Conv2d_2_pointwise/weights
H:F` (28MobilenetV1/Conv2d_2_pointwise/BatchNorm/moving_variance
=:;?2.MobilenetV1/Conv2d_5_depthwise/BatchNorm/gamma
L:J?21MobilenetV1/Conv2d_10_depthwise/depthwise_weights
>:<?2/MobilenetV1/Conv2d_11_pointwise/BatchNorm/gamma
=:;?2.MobilenetV1/Conv2d_6_pointwise/BatchNorm/gamma
?2?
__inference_call_fn_6108
__inference_call_fn_6418
__inference_call_fn_6758
__inference_call_fn_7068?
???
FullArgSpecL
argsD?A
jinputs

jtraining
jreturn_endpoints
jbatch_norm_momentum
varargs
 
varkw
 "
defaults?
p 
p 
	Y?G?z???

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference_<lambda>_7347?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
"__inference_signature_wrapper_5765inputs"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
__inference_<lambda>_7347??b?TR?"K)*SJ(f3|?U}4?z=o?[#5LDcpg{Pq:?+Ie'?,hr6/ ?!s_>`?GHxX-YE?n<	&ZVit0udM?~$@1Q7NA??mj?
W?8\yOB?2vC.k;9]^awFl%?

? 
? "? ?
__inference_call_fn_6108??b?TR?"K)*SJ(f3|?U}4?z=o?[#5LDcpg{Pq:?+Ie'?,hr6/ ?!s_>`?GHxX-YE?n<	&ZVit0udM?~$@1Q7NA??mj?
W?8\yOB?2vC.k;9]^awFl%o?l
e?b
:?7
inputs+???????????????????????????
p
p
?
batch_norm_momentum 
? "???
`
MobilenetV1/Conv2d_0H?E
MobilenetV1/Conv2d_0+???????????????????????????
w
MobilenetV1/Conv2d_10_depthwiseT?Q
MobilenetV1/Conv2d_10_depthwise,????????????????????????????
w
MobilenetV1/Conv2d_10_pointwiseT?Q
MobilenetV1/Conv2d_10_pointwise,????????????????????????????
w
MobilenetV1/Conv2d_11_depthwiseT?Q
MobilenetV1/Conv2d_11_depthwise,????????????????????????????
w
MobilenetV1/Conv2d_11_pointwiseT?Q
MobilenetV1/Conv2d_11_pointwise,????????????????????????????
w
MobilenetV1/Conv2d_12_depthwiseT?Q
MobilenetV1/Conv2d_12_depthwise,????????????????????????????
w
MobilenetV1/Conv2d_12_pointwiseT?Q
MobilenetV1/Conv2d_12_pointwise,????????????????????????????
w
MobilenetV1/Conv2d_13_depthwiseT?Q
MobilenetV1/Conv2d_13_depthwise,????????????????????????????
w
MobilenetV1/Conv2d_13_pointwiseT?Q
MobilenetV1/Conv2d_13_pointwise,????????????????????????????
t
MobilenetV1/Conv2d_1_depthwiseR?O
MobilenetV1/Conv2d_1_depthwise+???????????????????????????
t
MobilenetV1/Conv2d_1_pointwiseR?O
MobilenetV1/Conv2d_1_pointwise+???????????????????????????0
t
MobilenetV1/Conv2d_2_depthwiseR?O
MobilenetV1/Conv2d_2_depthwise+???????????????????????????0
t
MobilenetV1/Conv2d_2_pointwiseR?O
MobilenetV1/Conv2d_2_pointwise+???????????????????????????`
t
MobilenetV1/Conv2d_3_depthwiseR?O
MobilenetV1/Conv2d_3_depthwise+???????????????????????????`
t
MobilenetV1/Conv2d_3_pointwiseR?O
MobilenetV1/Conv2d_3_pointwise+???????????????????????????`
t
MobilenetV1/Conv2d_4_depthwiseR?O
MobilenetV1/Conv2d_4_depthwise+???????????????????????????`
u
MobilenetV1/Conv2d_4_pointwiseS?P
MobilenetV1/Conv2d_4_pointwise,????????????????????????????
u
MobilenetV1/Conv2d_5_depthwiseS?P
MobilenetV1/Conv2d_5_depthwise,????????????????????????????
u
MobilenetV1/Conv2d_5_pointwiseS?P
MobilenetV1/Conv2d_5_pointwise,????????????????????????????
u
MobilenetV1/Conv2d_6_depthwiseS?P
MobilenetV1/Conv2d_6_depthwise,????????????????????????????
u
MobilenetV1/Conv2d_6_pointwiseS?P
MobilenetV1/Conv2d_6_pointwise,????????????????????????????
u
MobilenetV1/Conv2d_7_depthwiseS?P
MobilenetV1/Conv2d_7_depthwise,????????????????????????????
u
MobilenetV1/Conv2d_7_pointwiseS?P
MobilenetV1/Conv2d_7_pointwise,????????????????????????????
u
MobilenetV1/Conv2d_8_depthwiseS?P
MobilenetV1/Conv2d_8_depthwise,????????????????????????????
u
MobilenetV1/Conv2d_8_pointwiseS?P
MobilenetV1/Conv2d_8_pointwise,????????????????????????????
u
MobilenetV1/Conv2d_9_depthwiseS?P
MobilenetV1/Conv2d_9_depthwise,????????????????????????????
u
MobilenetV1/Conv2d_9_pointwiseS?P
MobilenetV1/Conv2d_9_pointwise,????????????????????????????
C
MobilenetV1/Logits-?*
MobilenetV1/Logits??????????
M
MobilenetV1/Predictions2?/
MobilenetV1/Predictions??????????
U
MobilenetV1/global_pool:?7
MobilenetV1/global_pool??????????
-
default"?
default???????????
__inference_call_fn_6418??b?TR?"K)*SJ(f3|?U}4?z=o?[#5LDcpg{Pq:?+Ie'?,hr6/ ?!s_>`?GHxX-YE?n<	&ZVit0udM?~$@1Q7NA??mj?
W?8\yOB?2vC.k;9]^awFl%o?l
e?b
:?7
inputs+???????????????????????????
p
p 
?
batch_norm_momentum 
? "????????????
__inference_call_fn_6758??b?TR?"K)*SJ(f3|?U}4?z=o?[#5LDcpg{Pq:?+Ie'?,hr6/ ?!s_>`?GHxX-YE?n<	&ZVit0udM?~$@1Q7NA??mj?
W?8\yOB?2vC.k;9]^awFl%o?l
e?b
:?7
inputs+???????????????????????????
p 
p
?
batch_norm_momentum 
? "???
`
MobilenetV1/Conv2d_0H?E
MobilenetV1/Conv2d_0+???????????????????????????
w
MobilenetV1/Conv2d_10_depthwiseT?Q
MobilenetV1/Conv2d_10_depthwise,????????????????????????????
w
MobilenetV1/Conv2d_10_pointwiseT?Q
MobilenetV1/Conv2d_10_pointwise,????????????????????????????
w
MobilenetV1/Conv2d_11_depthwiseT?Q
MobilenetV1/Conv2d_11_depthwise,????????????????????????????
w
MobilenetV1/Conv2d_11_pointwiseT?Q
MobilenetV1/Conv2d_11_pointwise,????????????????????????????
w
MobilenetV1/Conv2d_12_depthwiseT?Q
MobilenetV1/Conv2d_12_depthwise,????????????????????????????
w
MobilenetV1/Conv2d_12_pointwiseT?Q
MobilenetV1/Conv2d_12_pointwise,????????????????????????????
w
MobilenetV1/Conv2d_13_depthwiseT?Q
MobilenetV1/Conv2d_13_depthwise,????????????????????????????
w
MobilenetV1/Conv2d_13_pointwiseT?Q
MobilenetV1/Conv2d_13_pointwise,????????????????????????????
t
MobilenetV1/Conv2d_1_depthwiseR?O
MobilenetV1/Conv2d_1_depthwise+???????????????????????????
t
MobilenetV1/Conv2d_1_pointwiseR?O
MobilenetV1/Conv2d_1_pointwise+???????????????????????????0
t
MobilenetV1/Conv2d_2_depthwiseR?O
MobilenetV1/Conv2d_2_depthwise+???????????????????????????0
t
MobilenetV1/Conv2d_2_pointwiseR?O
MobilenetV1/Conv2d_2_pointwise+???????????????????????????`
t
MobilenetV1/Conv2d_3_depthwiseR?O
MobilenetV1/Conv2d_3_depthwise+???????????????????????????`
t
MobilenetV1/Conv2d_3_pointwiseR?O
MobilenetV1/Conv2d_3_pointwise+???????????????????????????`
t
MobilenetV1/Conv2d_4_depthwiseR?O
MobilenetV1/Conv2d_4_depthwise+???????????????????????????`
u
MobilenetV1/Conv2d_4_pointwiseS?P
MobilenetV1/Conv2d_4_pointwise,????????????????????????????
u
MobilenetV1/Conv2d_5_depthwiseS?P
MobilenetV1/Conv2d_5_depthwise,????????????????????????????
u
MobilenetV1/Conv2d_5_pointwiseS?P
MobilenetV1/Conv2d_5_pointwise,????????????????????????????
u
MobilenetV1/Conv2d_6_depthwiseS?P
MobilenetV1/Conv2d_6_depthwise,????????????????????????????
u
MobilenetV1/Conv2d_6_pointwiseS?P
MobilenetV1/Conv2d_6_pointwise,????????????????????????????
u
MobilenetV1/Conv2d_7_depthwiseS?P
MobilenetV1/Conv2d_7_depthwise,????????????????????????????
u
MobilenetV1/Conv2d_7_pointwiseS?P
MobilenetV1/Conv2d_7_pointwise,????????????????????????????
u
MobilenetV1/Conv2d_8_depthwiseS?P
MobilenetV1/Conv2d_8_depthwise,????????????????????????????
u
MobilenetV1/Conv2d_8_pointwiseS?P
MobilenetV1/Conv2d_8_pointwise,????????????????????????????
u
MobilenetV1/Conv2d_9_depthwiseS?P
MobilenetV1/Conv2d_9_depthwise,????????????????????????????
u
MobilenetV1/Conv2d_9_pointwiseS?P
MobilenetV1/Conv2d_9_pointwise,????????????????????????????
C
MobilenetV1/Logits-?*
MobilenetV1/Logits??????????
M
MobilenetV1/Predictions2?/
MobilenetV1/Predictions??????????
U
MobilenetV1/global_pool:?7
MobilenetV1/global_pool??????????
-
default"?
default???????????
__inference_call_fn_7068??b?TR?"K)*SJ(f3|?U}4?z=o?[#5LDcpg{Pq:?+Ie'?,hr6/ ?!s_>`?GHxX-YE?n<	&ZVit0udM?~$@1Q7NA??mj?
W?8\yOB?2vC.k;9]^awFl%o?l
e?b
:?7
inputs+???????????????????????????
p 
p 
?
batch_norm_momentum 
? "????????????
"__inference_signature_wrapper_5765??b?TR?"K)*SJ(f3|?U}4?z=o?[#5LDcpg{Pq:?+Ie'?,hr6/ ?!s_>`?GHxX-YE?n<	&ZVit0udM?~$@1Q7NA??mj?
W?8\yOB?2vC.k;9]^awFl%S?P
? 
I?F
D
inputs:?7
inputs+???????????????????????????"0?-
+
logits!?
logits??????????