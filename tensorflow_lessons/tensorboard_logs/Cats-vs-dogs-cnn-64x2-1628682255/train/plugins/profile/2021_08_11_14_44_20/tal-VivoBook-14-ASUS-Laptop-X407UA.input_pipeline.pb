	??o??lK@??o??lK@!??o??lK@	???XP?????XP??!???XP??"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$??o??lK@ByGsd??AIJzZQK@YM?]~??*	???Mb V@2U
Iterator::Model::ParallelMapV2??+?S??!"?Qq?6@)??+?S??1"?Qq?6@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?e3????!ES=Q??<@)?7??w???1??Y2 6@:Preprocessing2F
Iterator::Model?? ??	??!??,5D<F@)?&?????1V=??5@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate6????$??!?[*e3@)???"???1??kJX?%@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceCqǛ?}?!????# @)CqǛ?}?1????# @:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorԷ?鲘x?!!F??(K@)Է?鲘x?1!F??(K@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip???[??!HP?ʻ?K@)ϻ??0(s?1a????A@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?O?c*??!????C5@)?C3O?)`?1?g?be?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9???XP??I???{?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	ByGsd??ByGsd??!ByGsd??      ??!       "      ??!       *      ??!       2	IJzZQK@IJzZQK@!IJzZQK@:      ??!       B      ??!       J	M?]~??M?]~??!M?]~??R      ??!       Z	M?]~??M?]~??!M?]~??b      ??!       JCPU_ONLYY???XP??b q???{?X@