from yolov3 import parse_cfg_list


def test_parse_cfg_list():
    lst = ['[some_type_1]', 'attribute_1=1', '[some_type_2]', 'attribute_2=2']
    blocks = parse_cfg_list(lst)
    print(blocks)
    # [{'type': 'some_type_1', 'attribute_1': '1'}, {'type': 'some_type_2', 'attribute_2': '2'}]
    assert isinstance(blocks,list)
    assert len(blocks) == 2
    assert isinstance(blocks[0],dict)
    assert len(blocks[0]) == 2
    assert blocks[0]['type'] == 'some_type_1'
    assert blocks[0]['attribute_1'] == '1'
    assert isinstance(blocks[1],dict)
    assert len(blocks[1]) == 2
    assert blocks[1]['type'] == 'some_type_2'
    assert blocks[1]['attribute_2'] == '2'
