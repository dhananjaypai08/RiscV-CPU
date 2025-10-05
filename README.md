# RISC-V ISA Implementation Reference

## To Implement 
**Note** : You can add your own program instructions in `main` function or can just run 
```sh
cargo run -p emulator # runs the interpreter with a fixed program
```

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

## Assembly to Machine Code Conversion Flow

This section explains the complete process of converting RISC-V assembly instructions to machine code and vice versa, with detailed mathematical breakdowns.

### Understanding RISC-V Instruction Formats

RISC-V uses different instruction formats for different types of operations:

#### **I-Type Format (Immediate Instructions)**
```
31    20 19 15 14 12 11  7 6   0
+-------+----+----+----+------+
|  imm  |rs1 |f3 | rd |opcode|
+-------+----+----+----+------+
 12 bits 5b  3b  5b  7 bits
```

#### **R-Type Format (Register-Register Instructions)**
```
31    25 24 20 19 15 14 12 11  7 6   0
+-------+----+----+----+----+------+
| funct7| rs2| rs1|f3 | rd |opcode|
+-------+----+----+----+----+------+
  7 bits 5b  5b  3b  5b  7 bits
```

### RISC-V Opcode and Function Code Tables

#### **Primary Opcodes (7-bit)**
| Instruction Type | Opcode (binary) | Opcode (hex) | Examples |
|------------------|-----------------|--------------|----------|
| OP-IMM           | 0010011         | 0x13         | addi, slti, xori |
| OP (Register)    | 0110011         | 0x33         | add, sub, xor |
| LOAD             | 0000011         | 0x03         | lw, lb, lh |
| STORE            | 0100011         | 0x23         | sw, sb, sh |
| BRANCH           | 1100011         | 0x63         | beq, bne, blt |

#### **OP-IMM funct3 Codes (I-type)**
| Instruction | funct3 (binary) | funct3 (decimal) |
|-------------|-----------------|------------------|
| addi        | 000             | 0                |
| slti        | 010             | 2                |
| sltiu       | 011             | 3                |
| xori        | 100             | 4                |
| ori         | 110             | 6                |
| andi        | 111             | 7                |

#### **OP funct3 and funct7 Codes (R-type)**
| Instruction | funct3 (binary) | funct7 (binary) |
|-------------|-----------------|-----------------|
| add         | 000             | 0000000         |
| sub         | 000             | 0100000         |
| sll         | 001             | 0000000         |
| slt         | 010             | 0000000         |
| sltu        | 011             | 0000000         |
| xor         | 100             | 0000000         |
| srl         | 101             | 0000000         |
| sra         | 101             | 0100000         |
| or          | 110             | 0000000         |
| and         | 111             | 0000000         |

### Detailed Conversion Example 1: I-Type Instruction

**Assembly:** `addi x3, x0, 21`

#### Step 1: Parse Assembly Components
```
addi x3, x0, 21
 |   |   |   |
 |   |   |   └── immediate value = 21
 |   |   └────── source register = x0 = register 0
 |   └────────── destination register = x3 = register 3
 └────────────── instruction mnemonic = addi
```

#### Step 2: Look Up Instruction Encoding
```
Instruction: addi
- Type: I-type (immediate)
- opcode: 0010011 (from OP-IMM table) = 0x13
- funct3: 000 (from addi table) = 0
```

#### Step 3: Convert Values to Binary Fields
```
rd     = x3 = 3    = 00011 (5 bits)
rs1    = x0 = 0    = 00000 (5 bits)
funct3 = addi      = 000 (3 bits) [looked up from table]
opcode = OP-IMM    = 0010011 (7 bits) [looked up from table]
imm    = 21        = 000000010101 (12 bits)
```

#### Step 4: Arrange Fields in I-Type Format
```
31    20 19 15 14 12 11  7 6   0
000000010101 00000 000 00011 0010011
    21        x0   add  x3   OP-IMM
```

#### Step 5: Convert to Single Binary Number
```
00000001010100000000001110010011
```

#### Step 6: Convert Binary to Hexadecimal
```
Group into 4-bit chunks:
0000 0001 0101 0000 0000 0001 1001 0011
  0    1    5    0    0    1    9    3
= 0x01500193
```

#### Step 7: Little-Endian Byte Storage
```
0x01500193 splits into bytes: [01] [50] [01] [93]

RISC-V uses little-endian storage (least significant byte first):
Memory layout: [0x93] [0x01] [0x50] [0x01]
```

#### Step 8: Byte Field Breakdown
```
Your program bytes: 0x93, 0x01, 0x50, 0x01

Byte 0: 0x93 = 10010011
  bits 6:0   = 0010011 = opcode (OP-IMM)
  bit 7      = 1 = rd[0] (part of x3)

Byte 1: 0x01 = 00000001
  bits 3:0   = 0001 = rd[4:1] (completes x3 = 00011)
  bits 6:4   = 000 = funct3 (ADDI)
  bit 7      = 0 = rs1[0] (part of x0)

Byte 2: 0x50 = 01010000
  bits 3:0   = 0000 = rs1[4:1] (completes x0 = 00000)
  bits 7:4   = 0101 = imm[3:0] (part of 21)

Byte 3: 0x01 = 00000001
  bits 7:0   = 00000001 = imm[11:4] (completes 21 = 000000010101)
```

### Detailed Conversion Example 2: R-Type Instruction

**Assembly:** `add x4, x2, x3`

#### Step 1: Parse Assembly Components
```
add x4, x2, x3
 |  |   |   |
 |  |   |   └── source register 2 = x3 = register 3
 |  |   └────── source register 1 = x2 = register 2
 |  └────────── destination register = x4 = register 4
 └───────────── instruction mnemonic = add
```

#### Step 2: Look Up Instruction Encoding
```
Instruction: add
- Type: R-type (register-register)
- opcode: 0110011 (from OP table) = 0x33
- funct3: 000 (from add table) = 0
- funct7: 0000000 (from add table) = 0
```

#### Step 3: Convert Values to Binary Fields
```
rd     = x4 = 4    = 00100 (5 bits)
rs1    = x2 = 2    = 00010 (5 bits)
rs2    = x3 = 3    = 00011 (5 bits)
funct3 = add       = 000 (3 bits) [looked up from table]
funct7 = add       = 0000000 (7 bits) [looked up from table]
opcode = OP        = 0110011 (7 bits) [looked up from table]
```

#### Step 4: Arrange Fields in R-Type Format
```
31    25 24 20 19 15 14 12 11  7 6   0
0000000 00011 00010 000 00100 0110011
   0     x3    x2   add  x4    OP
```

#### Step 5: Convert to Single Binary Number
```
00000000001100010000001000110011
```

#### Step 6: Convert Binary to Hexadecimal
```
Group into 4-bit chunks:
0000 0000 0011 0001 0000 0010 0011 0011
  0    0    3    1    0    2    3    3
= 0x00310233
```

#### Step 7: Little-Endian Byte Storage
```
0x00310233 splits into bytes: [00] [31] [02] [33]

Little-endian storage:
Memory layout: [0x33] [0x02] [0x31] [0x00]
```

#### Step 8: Byte Field Breakdown
```
Your program bytes: 0x33, 0x02, 0x31, 0x00

Byte 0: 0x33 = 00110011
  bits 6:0   = 0110011 = opcode (OP)
  bit 7      = 0 = rd[0] (part of x4)

Byte 1: 0x02 = 00000010
  bits 3:0   = 0010 = rd[4:1] (completes x4 = 00100)
  bits 6:4   = 000 = funct3 (ADD)
  bit 7      = 0 = rs1[0] (part of x2)

Byte 2: 0x31 = 00110001
  bits 3:0   = 0001 = rs1[4:1] (completes x2 = 00010)
  bits 7:4   = 0011 = rs2[3:0] (part of x3)

Byte 3: 0x00 = 00000000
  bit 0      = 0 = rs2[4] (completes x3 = 00011)
  bits 7:1   = 0000000 = funct7 (ADD)
```

### Reverse Engineering: Machine Code to Assembly

To convert machine code back to assembly, reverse the process:

#### For bytes `0x93, 0x01, 0x50, 0x01`:

1. **Combine to 32-bit word (little-endian):**
   ```
   0x01500193
   ```

2. **Extract fields using bit masks:**
   ```
   opcode = instruction & 0x7F          = 0x13 (OP-IMM)
   rd     = (instruction >> 7) & 0x1F   = 3 (x3)
   funct3 = (instruction >> 12) & 0x07  = 0 (ADDI)
   rs1    = (instruction >> 15) & 0x1F  = 0 (x0)
   imm    = (instruction >> 20) & 0xFFF = 21
   ```

3. **Decode instruction:**
   ```
   opcode 0x13 + funct3 0 = addi
   Result: addi x3, x0, 21
   ```

### Key Mathematical Concepts

#### **Bit Manipulation Operations**
- **AND (&):** Used to extract specific bits: `value & mask`
- **OR (|):** Used to combine bit fields: `field1 | field2`
- **Shift Left (<<):** Moves bits left: `value << 3` multiplies by 8
- **Shift Right (>>):** Moves bits right: `value >> 3` divides by 8

#### **Little-Endian Storage**
- **Concept:** Least significant byte stored at lowest memory address
- **32-bit word 0x12345678:** Stored as bytes [0x78, 0x56, 0x34, 0x12]
- **Why important:** CPU architecture determines byte ordering in memory

#### **Sign Extension**
- **Purpose:** Extend smaller signed numbers to larger bit widths
- **Method:** If MSB = 1, fill upper bits with 1s; if MSB = 0, fill with 0s
- **Example:** 12-bit immediate -1 (0xFFF) extends to 0xFFFFFFFF in 32-bit

#### **Two's Complement Arithmetic**
- **Negative numbers:** Invert all bits and add 1
- **Range:** n-bit signed number ranges from -2^(n-1) to 2^(n-1)-1
- **Example:** 12-bit immediate range: -2048 to +2047

### Instruction Execution Flow

1. **Fetch:** CPU reads 4 bytes from memory at PC address
2. **Decode:** Extract opcode, determine instruction type, extract fields
3. **Execute:** Perform operation based on instruction type
4. **Writeback:** Store result in destination register
5. **Update PC:** Advance program counter to next instruction

This systematic approach ensures every RISC-V instruction follows the same encoding and decoding principles, making the ISA both regular and predictable.