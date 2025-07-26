# RISC-V ISA Implementation Reference

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