import torch
import pytest
from quick_model_loader.allocator import (
    MemoryTable,
    get_insert_index,
    find_feasible_addr,
)
from quick_model_loader.meta import SliceInfo


def test_get_insert_index():
    slice_info0 = SliceInfo(16, 28)
    slice_info1 = SliceInfo(48, 52)
    table = [slice_info0, slice_info1]

    slice_info_illegal = SliceInfo(0, 40)
    index = get_insert_index(table, slice_info_illegal)
    assert index is None

    slice_info2 = SliceInfo(0, 8)
    index = get_insert_index(table, slice_info2)
    assert index == 0

    slice_info3 = SliceInfo(32, 36)
    index = get_insert_index(table, slice_info3)
    assert index == 1


def test_append():
    memory_table = MemoryTable(64)

    slice_info_illegal = SliceInfo(-16, 0)
    with pytest.raises(
        MemoryError, match="Illegal memory allocation! Memory address should >= 0!"
    ):
        memory_table.append(slice_info_illegal)

    slice_info_illegal = SliceInfo(48, 72)
    with pytest.raises(
        MemoryError, match="Illegal memory allocation! Exceed memory size!"
    ):
        memory_table.append(slice_info_illegal)

    slice_info_illegal = SliceInfo(2, 10)
    with pytest.raises(
        MemoryError, match="Illegal memory allocation! Memory not alligned!"
    ):
        memory_table.append(slice_info_illegal)

    slice_info0 = SliceInfo(16, 20)
    memory_table.append(slice_info0)
    assert len(memory_table) == 1
    assert memory_table[0] == slice_info0

    slice_info1 = SliceInfo(0, 4)
    memory_table.append(slice_info1)
    assert len(memory_table) == 2
    assert memory_table[0] == slice_info1
    assert memory_table[1] == slice_info0

    slice_info_illegal = SliceInfo(0, 32)
    with pytest.raises(
        MemoryError, match="Illegal memory allocation! Memory has been allocated!"
    ):
        memory_table.append(slice_info_illegal)

    slice_info2 = SliceInfo(48, 54)
    slice_info3 = SliceInfo(32, 40)
    memory_table.append(slice_info3)
    memory_table.append(slice_info2)

    assert len(memory_table) == 4
    assert memory_table[0] == slice_info1
    assert memory_table[1] == slice_info0
    assert memory_table[2] == slice_info3
    assert memory_table[3] == slice_info2


def test_search_allocated():
    memory_table = MemoryTable(64)

    slice_info0 = SliceInfo(16, 20)
    slice_info1 = SliceInfo(48, 54)
    slice_info2 = SliceInfo(0, 4)
    slice_info3 = SliceInfo(32, 40)

    memory_table.append(slice_info0)
    memory_table.append(slice_info1)
    memory_table.append(slice_info2)
    memory_table.append(slice_info3)

    index = memory_table.search_allocated(slice_info0)
    assert index == 1
    index = memory_table.search_allocated(slice_info1)
    assert index == 3
    index = memory_table.search_allocated(slice_info2)
    assert index == 0
    index = memory_table.search_allocated(slice_info3)
    assert index == 2

    slice_info_illegal = SliceInfo(32, 54)
    index = memory_table.search_allocated(slice_info_illegal)
    assert index is None


def test_delete():
    memory_table = MemoryTable(96)

    slice_info0 = SliceInfo(16, 20)
    slice_info1 = SliceInfo(48, 54)
    slice_info2 = SliceInfo(0, 4)
    slice_info3 = SliceInfo(32, 40)

    memory_table.append(slice_info0)
    memory_table.append(slice_info1)
    memory_table.append(slice_info2)
    memory_table.append(slice_info3)

    assert len(memory_table) == 4

    slice_info_illegal = SliceInfo(80, 96)
    with pytest.raises(MemoryError, match="Try to free unallocated memory!"):
        memory_table.delete(slice_info_illegal)

    memory_table.delete(slice_info0)
    assert len(memory_table) == 3
    assert memory_table[0] == slice_info2
    assert memory_table[1] == slice_info3
    assert memory_table[2] == slice_info1

    memory_table.delete(slice_info1)
    assert len(memory_table) == 2
    assert memory_table[0] == slice_info2
    assert memory_table[1] == slice_info3

    memory_table.delete(slice_info2)
    assert len(memory_table) == 1
    assert memory_table[0] == slice_info3

    memory_table.delete(slice_info3)
    assert len(memory_table) == 0


def test_find_feasible_addr():
    slice_size0 = 4

    addr = find_feasible_addr(0, 4, slice_size0)
    assert addr == 0
    addr = find_feasible_addr(2, 6, slice_size0)
    assert addr is None
    addr = find_feasible_addr(2, 8, slice_size0)
    assert addr is None

    slice_size0 = 8
    addr = find_feasible_addr(16, 40, slice_size0)
    assert addr == 16
    addr = find_feasible_addr(20, 30, slice_size0)
    assert addr is None
    addr = find_feasible_addr(32, 40, slice_size0)
    assert addr == 32


def test_find_free_addr():
    memory_table = MemoryTable(80)
    slice_info0 = SliceInfo(0, 16)
    slice_info1 = SliceInfo(32, 40)
    memory_table.append(slice_info0)
    memory_table.append(slice_info1)

    slice_size = 12
    addr = memory_table.find_free_addr(slice_size)
    assert addr == 16

    slice_size = 10
    addr = memory_table.find_free_addr(slice_size)
    assert addr == 16

    slice_size = 6
    addr = memory_table.find_free_addr(slice_size)
    assert addr == 16

    slice_size = 24
    addr = memory_table.find_free_addr(slice_size)
    assert addr == 48

    slice_size = 36
    addr = memory_table.find_free_addr(slice_size)
    assert addr is None


def test_get_allocated_size():
    memory_table = MemoryTable(64)

    slice_info0 = SliceInfo(16, 20)
    slice_info1 = SliceInfo(48, 54)
    slice_info2 = SliceInfo(0, 4)
    slice_info3 = SliceInfo(32, 40)

    memory_table.append(slice_info0)
    assert memory_table.get_allocated_size() == 4
    assert memory_table.get_free_size() == 60

    memory_table.append(slice_info1)
    assert memory_table.get_allocated_size() == 10
    assert memory_table.get_free_size() == 54

    memory_table.append(slice_info2)
    assert memory_table.get_allocated_size() == 14
    assert memory_table.get_free_size() == 50

    memory_table.append(slice_info3)
    assert memory_table.get_allocated_size() == 22
    assert memory_table.get_free_size() == 42


def test_can_allocate_when_empty():
    memory_table = MemoryTable(80)
    slice_size0 = 18
    slice_size1 = 16
    slice_size2 = 24
    slice_size3 = 6

    assert not memory_table.can_allocate_when_empty(
        [slice_size0, slice_size1, slice_size2, slice_size3]
    )

    slice_size2 = 12

    assert memory_table.can_allocate_when_empty(
        [slice_size0, slice_size1, slice_size2, slice_size3]
    )


def test_can_allocate():
    memory_table = MemoryTable(80)
    slice_size0 = 18
    slice_size1 = 16
    slice_size2 = 24
    slice_size4 = 6

    assert not memory_table.can_allocate(
        [slice_size0, slice_size1, slice_size2, slice_size4]
    )

    slice_size2 = 12

    assert memory_table.can_allocate(
        [slice_size0, slice_size1, slice_size2, slice_size4]
    )

    slice_info0 = SliceInfo(0, 18)
    slice_info1 = SliceInfo(32, 48)
    memory_table.append(slice_info0)
    memory_table.append(slice_info1)

    slice_size0 = 12
    slice_size1 = 24

    assert not memory_table.can_allocate([slice_size0, slice_size1])

    slice_size1 = 12
    assert memory_table.can_allocate([slice_size0, slice_size1])

    memory_table.clear()
    slice_info0 = SliceInfo(0, 18)
    slice_info1 = SliceInfo(48, 50)
    memory_table.append(slice_info0)
    memory_table.append(slice_info1)

    slice_size0 = 18
    slice_size1 = 24
    assert not memory_table.can_allocate([slice_size0, slice_size1])

    slice_size0 = 18
    slice_size1 = 16
    assert not memory_table.can_allocate([slice_size0, slice_size1])

    slice_size0 = 18
    slice_size1 = 8
    assert not memory_table.can_allocate([slice_size0, slice_size1])

    slice_size0 = 16
    slice_size1 = 8
    assert memory_table.can_allocate([slice_size0, slice_size1])


def test_can_move():
    memory_table = MemoryTable(80)

    slice_info0 = SliceInfo(16, 60)
    memory_table.append(slice_info0)

    assert not memory_table.can_move(slice_info0, 2)
    assert memory_table.can_move(slice_info0, 0)

    assert not memory_table.can_move(slice_info0, 6)
    assert not memory_table.can_move(slice_info0, 12)
    assert memory_table.can_move(slice_info0, 32)

    memory_table.clear()
    slice_info0 = SliceInfo(0, 16)
    slice_info1 = SliceInfo(32, 48)
    memory_table.append(slice_info0)
    memory_table.append(slice_info1)

    assert not memory_table.can_move(slice_info0, 24)
    assert not memory_table.can_move(slice_info0, 44)
    assert memory_table.can_move(slice_info0, 64)
    assert memory_table.can_move(slice_info0, 48)


def test_move():
    memory_table = MemoryTable(72)

    slice_info0 = SliceInfo(16, 60)
    memory_table.append(slice_info0)

    with pytest.raises(RuntimeError, match="Illegal slice movement!"):
        memory_table.move(slice_info0, 2)

    with pytest.raises(RuntimeError, match="Illegal slice movement!"):
        memory_table.move(slice_info0, 6)

    with pytest.raises(RuntimeError, match="Illegal slice movement!"):
        memory_table.move(slice_info0, 12)

    with pytest.raises(RuntimeError, match="Illegal slice movement!"):
        memory_table.move(slice_info0, 32)

    memory_table.move(slice_info0, 0)
    assert len(memory_table) == 1
    assert memory_table[0] == SliceInfo(0, 44)

    memory_table.move(slice_info0, 16)
    assert len(memory_table) == 1
    assert memory_table[0] == SliceInfo(16, 60)

    memory_table.clear()
    slice_info0 = SliceInfo(0, 18)
    slice_info1 = SliceInfo(32, 48)
    memory_table.append(slice_info0)
    memory_table.append(slice_info1)

    with pytest.raises(RuntimeError, match="Illegal slice movement!"):
        memory_table.move(slice_info0, 16)

    with pytest.raises(RuntimeError, match="Illegal slice movement!"):
        memory_table.move(slice_info0, 32)

    memory_table.move(slice_info0, 48)
    assert len(memory_table) == 2
    assert memory_table[0] == SliceInfo(32, 48)
    assert memory_table[1] == SliceInfo(48, 66)

    with pytest.raises(RuntimeError, match="Illegal slice movement!"):
        memory_table.move(slice_info0, 40)

    with pytest.raises(RuntimeError, match="Illegal slice movement!"):
        memory_table.move(slice_info0, 28)

    memory_table.move(slice_info1, 0)
    memory_table.move(slice_info0, 16)
    assert len(memory_table) == 2
    assert memory_table[0] == SliceInfo(0, 16)
    assert memory_table[1] == SliceInfo(16, 34)
