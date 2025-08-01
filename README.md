# RISC-V ISA Implementation Reference

## RISC-V Registers

| Register | ABI Name | Description         | Saved By |
|----------|----------|---------------------|----------|
| x0       | zero     | Hard-wired zero     | -        |
| x1       | ra       | Return address      | Caller   |
| x2       | sp       | Stack pointer       | Callee   |
| x3       | gp       | Global pointer      | -        |
| x4       | tp       | Thread pointer      | -        |
| x5-x7    | t0-t2    | Temporaries         | Caller   |
| x8       | s0/fp    | Saved/frame pointer | Callee   |
| x9       | s1       | Saved register      | Callee   |
| x10-x11  | a0-a1    | Function args/ret   | Caller   |
| x12-x17  | a2-a7    | Function args       | Caller   |
| x18-x27  | s2-s11   | Saved registers     | Callee   |
| x28-x31  | t3-t6    | Temporaries         | Caller   |

## Register Table Fields

- **rd**: Destination register (where result is written)
- **rs1**: Source register 1
- **rs2**: Source register 2
- **imm**: Immediate value (constant encoded in instruction)
- **opcode**: Operation code (defines instruction type)
- **funct3/funct7**: Further specify operation within opcode

## Hex to Binary Conversion

Each hex digit = 4 binary bits. Example:

`0xA` = `1010`
`0x2A` = `00101010`
`0x02A00113` = `0000 0010 1010 0000 0000 0001 0001 0011`

## Instruction Field Breakdown

### Example: `addi x1, x0, 42`

1. **Assembly:** `addi x1, x0, 42`
2. **I-type format:**

   | Bits   | Field   | Value (for this example) |
   |--------|---------|-------------------------|
   | 31-20  | imm     | 000000101010 (42)       |
   | 19-15  | rs1     | 00000 (x0)              |
   | 14-12  | funct3  | 000                     |
   | 11-7   | rd      | 00001 (x1)              |
   | 6-0    | opcode  | 0010011 (0x13)          |

3. **Binary:** `000000101010 00000 000 00001 0010011`
4. **Hex:** `0x02A00113`
5. **Little-endian bytes:** `0x13, 0x01, 0xA0, 0x02`

## RISC-V Instruction Formats (Summary)

### R-Type (Register-Register)
```
31    25 24  20 19  15 14   12 11   7 6     0
+-------+-----+-----+-------+-----+---------+
| funct7| rs2 | rs1 | funct3| rd  | opcode  |
+-------+-----+-----+-------+-----+---------+
   7      5     5      3      5       7
```
**Usage:** `add rd, rs1, rs2` - rd = rs1 + rs2

### I-Type (Immediate)
```
31          20 19  15 14   12 11   7 6     0
+-------------+-----+-------+-----+---------+
|    imm      | rs1 | funct3| rd  | opcode  |
+-------------+-----+-------+-----+---------+
     12         5      3      5       7
```
**Usage:** `addi rd, rs1, imm` - rd = rs1 + imm

### S-Type (Store)
```
31    25 24  20 19  15 14   12 11   7 6     0
+-------+-----+-----+-------+-----+---------+
|imm[11:5]| rs2 | rs1 | funct3|imm[4:0]| opcode |
+-------+-----+-----+-------+-----+---------+
```
**Usage:** `sw rs2, imm(rs1)` - Memory[rs1 + imm] = rs2

### B-Type (Branch)
```
31 30    25 24  20 19  15 14   12 11  8 7 6     0
+--+-------+-----+-----+-------+----+-+---------+
|imm[12]|imm[10:5]| rs2 | rs1 | funct3|imm[4:1]|imm[11]| opcode |
+--+-------+-----+-----+-------+----+-+---------+
```
**Usage:** `beq rs1, rs2, imm` - if (rs1 == rs2) PC += imm

### U-Type (Upper Immediate)
```
31                  12 11   7 6     0
+---------------------+-----+---------+
|       imm[31:12]    | rd  | opcode  |
+---------------------+-----+---------+
```
**Usage:** `lui rd, imm` - rd = imm << 12

### J-Type (Jump)
```
31 30      21 20 19     12 11   7 6     0
+--+---------+--+--------+-----+---------+
|imm[20]|imm[10:1]|imm[11]|imm[19:12]| rd | opcode |
+--+---------+--+--------+-----+---------+
```
**Usage:** `jal rd, imm` - rd = PC + 4; PC += imm

## Field Descriptions

- **opcode**: 7 bits, always at bits 0-6, determines instruction type
- **rd**: 5 bits, destination register (bits 7-11)
- **funct3**: 3 bits, further specifies operation (bits 12-14)
- **rs1**: 5 bits, source register 1 (bits 15-19)
- **rs2**: 5 bits, source register 2 (bits 20-24, R/S/B types)
- **funct7**: 7 bits, further specifies operation (bits 25-31, R-type)
- **imm**: Immediate value, size and position depends on format

## In-Depth: RISC-V Base Integer Instruction Set (RV32I/RV64I)

### Arithmetic/Logic
| Instruction | Format | Description |
|-------------|--------|-------------|
| add         | R      | rd = rs1 + rs2 |
| sub         | R      | rd = rs1 - rs2 |
| addi        | I      | rd = rs1 + imm |
| xor         | R      | rd = rs1 ^ rs2 |
| xori        | I      | rd = rs1 ^ imm |
| or          | R      | rd = rs1 \| rs2 |
| ori         | I      | rd = rs1 \| imm |
| and         | R      | rd = rs1 & rs2 |
| andi        | I      | rd = rs1 & imm |
| sll         | R      | rd = rs1 << (rs2 & 0x3f) |
| slli        | I      | rd = rs1 << shamt |
| srl         | R      | rd = rs1 >> (rs2 & 0x3f) |
| srli        | I      | rd = rs1 >> shamt |
| sra         | R      | rd = arithmetic right shift |
| srai        | I      | rd = arithmetic right shift |

### Loads/Stores
| Instruction | Format | Description |
|-------------|--------|-------------|
| lb/lh/lw/ld | I      | Load byte/half/word/dword |
| lbu/lhu/lwu | I      | Load unsigned byte/half/word |
| sb/sh/sw/sd | S      | Store byte/half/word/dword |

### Branches/Jumps
| Instruction | Format | Description |
|-------------|--------|-------------|
| beq/bne     | B      | Branch if equal/not equal |
| blt/bge     | B      | Branch if less/greater-equal (signed) |
| bltu/bgeu   | B      | Branch if less/greater-equal (unsigned) |
| jal         | J      | Jump and link |
| jalr        | I      | Jump and link register |

### Upper Immediate
| Instruction | Format | Description |
|-------------|--------|-------------|
| lui         | U      | Load upper immediate |
| auipc       | U      | Add upper immediate to PC |

### System
| Instruction | Format | Description |
|-------------|--------|-------------|
| ecall/ebreak| I      | Environment call/break |

## Example: Full Encoding Walkthrough

### `addi x1, x0, 42` step-by-step

1. **Fields:**
   - opcode: 0010011 (0x13)
   - rd: 00001 (x1)
   - funct3: 000
   - rs1: 00000 (x0)
   - imm: 000000101010 (42)
2. **Binary:** `000000101010 00000 000 00001 0010011`
3. **Hex:** `0x02A00113`
4. **Little-endian bytes:** `0x13, 0x01, 0xA0, 0x02`
5. **In memory:**
   - Address 0: 0x13
   - Address 1: 0x01
   - Address 2: 0xA0
   - Address 3: 0x02
6. **CPU fetches 4 bytes, assembles to 0x02A00113, decodes fields, and executes the instruction.**

## Instruction Formats

### R-Type (Register-Register)
```
31    25 24  20 19  15 14   12 11   7 6     0
+-------+-----+-----+-------+-----+---------+
| funct7| rs2 | rs1 | funct3| rd  | opcode  |
+-------+-----+-----+-------+-----+---------+
   7      5     5      3      5       7
```
**Usage:** `add rd, rs1, rs2` - rd = rs1 + rs2

### I-Type (Immediate)
```
31          20 19  15 14   12 11   7 6     0
+-------------+-----+-------+-----+---------+
|    imm      | rs1 | funct3| rd  | opcode  |
+-------------+-----+-------+-----+---------+
     12         5      3      5       7
```
**Usage:** `addi rd, rs1, imm` - rd = rs1 + imm

### S-Type (Store)
```
31    25 24  20 19  15 14   12 11   7 6     0
+-------+-----+-----+-------+-----+---------+
|imm[11:5]| rs2 | rs1 | funct3|imm[4:0]| opcode |
+-------+-----+-----+-------+-----+---------+
```
**Usage:** `sw rs2, imm(rs1)` - Memory[rs1 + imm] = rs2

### B-Type (Branch)
```
31 30    25 24  20 19  15 14   12 11  8 7 6     0
+--+-------+-----+-----+-------+----+-+---------+
|imm[12]|imm[10:5]| rs2 | rs1 | funct3|imm[4:1]|imm[11]| opcode |
+--+-------+-----+-----+-------+----+-+---------+
```
**Usage:** `beq rs1, rs2, imm` - if (rs1 == rs2) PC += imm

### U-Type (Upper Immediate)
```
31                  12 11   7 6     0
+---------------------+-----+---------+
|       imm[31:12]    | rd  | opcode  |
+---------------------+-----+---------+
```
**Usage:** `lui rd, imm` - rd = imm << 12

### J-Type (Jump)
```
31 30      21 20 19     12 11   7 6     0
+--+---------+--+--------+-----+---------+
|imm[20]|imm[10:1]|imm[11]|imm[19:12]| rd | opcode |
+--+---------+--+--------+-----+---------+
```
**Usage:** `jal rd, imm` - rd = PC + 4; PC += imm

## Opcode Map

| Opcode    | Binary    | Instruction Type | Description |
|-----------|-----------|------------------|-------------|
| 0b0110111 | 0x37      | LUI              | Load Upper Immediate |
| 0b0010111 | 0x17      | AUIPC            | Add Upper Immediate to PC |
| 0b1101111 | 0x6F      | JAL              | Jump and Link |
| 0b1100111 | 0x67      | JALR             | Jump and Link Register |
| 0b1100011 | 0x63      | BRANCH           | Branch Instructions |
| 0b0000011 | 0x03      | LOAD             | Load Instructions |
| 0b0100011 | 0x23      | STORE            | Store Instructions |
| 0b0010011 | 0x13      | OP-IMM           | Immediate Operations |
| 0b0110011 | 0x33      | OP               | Register Operations |
| 0b0001111 | 0x0F      | MISC-MEM         | Memory Ordering |
| 0b1110011 | 0x73      | SYSTEM           | System Instructions |

## RV32M Extension (Multiplication/Division)

| Instruction | funct7   | funct3 | Description |
|-------------|----------|--------|-------------|
| MUL         | 0000001  | 000    | Multiply (lower 32 bits) |
| MULH        | 0000001  | 001    | Multiply High (signed × signed) |
| MULHSU      | 0000001  | 010    | Multiply High (signed × unsigned) |
| MULHU       | 0000001  | 011    | Multiply High (unsigned × unsigned) |
| DIV         | 0000001  | 100    | Divide (signed) |
| DIVU        | 0000001  | 101    | Divide (unsigned) |
| REM         | 0000001  | 110    | Remainder (signed) |
| REMU        | 0000001  | 111    | Remainder (unsigned) |

## Register Usage Conventions

| Register | ABI Name | Description | Saver |
|----------|----------|-------------|-------|
| x0       | zero     | Hard-wired zero | - |
| x1       | ra       | Return address | Caller |
| x2       | sp       | Stack pointer | Callee |
| x3       | gp       | Global pointer | - |
| x4       | tp       | Thread pointer | - |
| x5-7     | t0-2     | Temporaries | Caller |
| x8       | s0/fp    | Saved/frame pointer | Callee |
| x9       | s1       | Saved register | Callee |
| x10-11   | a0-1     | Function args/return | Caller |
| x12-17   | a2-7     | Function args | Caller |
| x18-27   | s2-11    | Saved registers | Callee |
| x28-31   | t3-6     | Temporaries | Caller |

## Memory Access Patterns

### Load Instructions (funct3)
- **000 (LB):** Load Byte (sign-extended)
- **001 (LH):** Load Halfword (sign-extended)
- **010 (LW):** Load Word (sign-extended)
- **011 (LD):** Load Doubleword (RV64 only)
- **100 (LBU):** Load Byte Unsigned
- **101 (LHU):** Load Halfword Unsigned
- **110 (LWU):** Load Word Unsigned (RV64 only)

### Store Instructions (funct3)
- **000 (SB):** Store Byte
- **001 (SH):** Store Halfword
- **010 (SW):** Store Word
- **011 (SD):** Store Doubleword (RV64 only)

### Branch Instructions (funct3)
- **000 (BEQ):** Branch if Equal
- **001 (BNE):** Branch if Not Equal
- **100 (BLT):** Branch if Less Than (signed)
- **101 (BGE):** Branch if Greater or Equal (signed)
- **110 (BLTU):** Branch if Less Than (unsigned)
- **111 (BGEU):** Branch if Greater or Equal (unsigned)