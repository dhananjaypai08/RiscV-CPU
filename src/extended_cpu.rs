use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct ExtendedCpu {
    pub registers: [u64; 32],
    pub pc: u64,
    pub memory: HashMap<u64, u8>,
    pub instruction_count: u64,
    pub cycle_count: u64,
    pub debug_mode: bool,
}

#[derive(Debug, Clone)]
pub struct ExecutionTrace {
    pub pc: u64,
    pub instruction: u32,
    pub opcode: String,
    pub registers_before: [u64; 32],
    pub registers_after: [u64; 32],
    pub memory_accesses: Vec<MemoryAccess>,
}

#[derive(Debug, Clone)]
pub struct MemoryAccess {
    pub address: u64,
    pub value: u64,
    pub size: usize,
    pub is_write: bool,
}

impl ExtendedCpu {
    pub fn new() -> Self {
        Self {
            registers: [0; 32],
            pc: 0,
            memory: HashMap::new(),
            instruction_count: 0,
            cycle_count: 0,
            debug_mode: false,
        }
    }

    pub fn enable_debug(&mut self) {
        self.debug_mode = true;
    }

    pub fn load_program(&mut self, program: &[u8], start_addr: u64) {
        for (i, &byte) in program.iter().enumerate() {
            self.memory.insert(start_addr + i as u64, byte);
        }
        self.pc = start_addr;
        if self.debug_mode {
            println!("Loaded {} bytes at 0x{:08x}", program.len(), start_addr);
        }
    }

    pub fn load_elf_section(&mut self, data: &[u8], vaddr: u64) {
        for (i, &byte) in data.iter().enumerate() {
            self.memory.insert(vaddr + i as u64, byte);
        }
        if self.debug_mode {
            println!("Loaded ELF section: {} bytes at 0x{:08x}", data.len(), vaddr);
        }
    }

    pub fn setup_stack(&mut self, stack_base: u64, stack_size: u64) {
        self.write_register(2, stack_base + stack_size);
        if self.debug_mode {
            println!("Stack setup: base=0x{:08x}, size={}, sp=0x{:08x}", 
                     stack_base, stack_size, stack_base + stack_size);
        }
    }

    pub fn read_memory(&mut self, addr: u64, bytes: usize) -> u64 {
        let mut result = 0u64;
        for i in 0..bytes {
            if let Some(&byte) = self.memory.get(&(addr + i as u64)) {
                result |= (byte as u64) << (i * 8);
            }
        }
        
        if self.debug_mode {
            println!("Memory read: addr=0x{:08x}, size={}, value=0x{:016x}", addr, bytes, result);
        }
        result
    }

    pub fn write_memory(&mut self, addr: u64, value: u64, bytes: usize) {
        for i in 0..bytes {
            let byte = ((value >> (i * 8)) & 0xff) as u8;
            self.memory.insert(addr + i as u64, byte);
        }
        
        if self.debug_mode {
            println!("Memory write: addr=0x{:08x}, size={}, value=0x{:016x}", addr, bytes, value);
        }
    }

    pub fn read_register(&self, reg: usize) -> u64 {
        if reg == 0 { 0 } else { self.registers[reg] }
    }

    pub fn write_register(&mut self, reg: usize, value: u64) {
        if reg != 0 {
            if self.debug_mode && reg != 0 {
                println!("Register write: x{} = 0x{:016x} (was 0x{:016x})", 
                         reg, value, self.registers[reg]);
            }
            self.registers[reg] = value;
        }
    }

    pub fn fetch_instruction(&mut self) -> u32 {
        let inst = self.read_memory(self.pc, 4) as u32;
        if self.debug_mode {
            println!("Fetch: PC=0x{:08x}, instruction=0x{:08x}", self.pc, inst);
        }
        inst
    }

    pub fn step(&mut self) -> Result<ExecutionTrace, String> {
        let registers_before = self.registers.clone();
        let pc_before = self.pc;
        let inst = self.fetch_instruction();
        
        let opcode_str = self.disassemble_instruction(inst);
        
        self.execute_instruction(inst)?;
        self.instruction_count += 1;
        self.cycle_count += 1;

        Ok(ExecutionTrace {
            pc: pc_before,
            instruction: inst,
            opcode: opcode_str,
            registers_before,
            registers_after: self.registers.clone(),
            memory_accesses: vec![], // Would be populated by memory operations
        })
    }

    pub fn execute_instruction(&mut self, inst: u32) -> Result<(), String> {
        let opcode = inst & 0x7f;
        let rd = ((inst >> 7) & 0x1f) as usize;
        let funct3 = (inst >> 12) & 0x07;
        let rs1 = ((inst >> 15) & 0x1f) as usize;
        let rs2 = ((inst >> 20) & 0x1f) as usize;
        let funct7 = (inst >> 25) & 0x7f;

        match opcode {
            0b0010011 => self.execute_op_imm(inst, rd, funct3, rs1),
            0b0110011 => self.execute_op(rd, funct3, rs1, rs2, funct7),
            0b0000011 => self.execute_load(rd, funct3, rs1, inst),
            0b0100011 => self.execute_store(funct3, rs1, rs2, inst),
            0b1100011 => self.execute_branch(funct3, rs1, rs2, inst),
            0b1101111 => self.execute_jal(rd, inst),
            0b1100111 => self.execute_jalr(rd, rs1, inst),
            0b0110111 => self.execute_lui(rd, inst),
            0b0010111 => self.execute_auipc(rd, inst),
            0b1110011 => self.execute_system(inst),
            0b0001111 => self.execute_fence(inst),
            _ => {
                if self.debug_mode {
                    println!("Unknown opcode: 0b{:07b} (0x{:08x})", opcode, inst);
                }
                self.pc += 4;
                Ok(())
            }
        }
    }

    fn execute_op_imm(&mut self, inst: u32, rd: usize, funct3: u32, rs1: usize) -> Result<(), String> {
        let imm = ((inst as i32) >> 20) as i64 as u64;
        let rs1_val = self.read_register(rs1);

        let result = match funct3 {
            0b000 => {
                // ADDI: Add immediate
                rs1_val.wrapping_add(imm)
            },
            0b010 => {
                // SLTI: Set less than immediate (signed)
                if (rs1_val as i64) < (imm as i64) { 1 } else { 0 }
            },
            0b011 => {
                // SLTIU: Set less than immediate (unsigned)
                if rs1_val < imm { 1 } else { 0 }
            },
            0b100 => {
                // XORI: XOR immediate
                rs1_val ^ imm
            },
            0b110 => {
                // ORI: OR immediate
                rs1_val | imm
            },
            0b111 => {
                // ANDI: AND immediate
                rs1_val & imm
            },
            0b001 => {
                // SLLI: Shift left logical immediate
                let shamt = (imm & 0x3f) as u32;
                if (inst >> 26) != 0 {
                    return Err("Invalid SLLI instruction".to_string());
                }
                rs1_val << shamt
            },
            0b101 => {
                // SRLI/SRAI: Shift right logical/arithmetic immediate
                let shamt = (imm & 0x3f) as u32;
                if (inst >> 30) & 1 == 0 {
                    // SRLI: Shift right logical
                    rs1_val >> shamt
                } else {
                    // SRAI: Shift right arithmetic
                    ((rs1_val as i64) >> shamt) as u64
                }
            },
            _ => return Err(format!("Invalid funct3 for OP-IMM: {}", funct3)),
        };

        self.write_register(rd, result);
        self.pc += 4;
        Ok(())
    }

    fn execute_op(&mut self, rd: usize, funct3: u32, rs1: usize, rs2: usize, funct7: u32) -> Result<(), String> {
        let rs1_val = self.read_register(rs1);
        let rs2_val = self.read_register(rs2);

        let result = match (funct7, funct3) {
            // RV32I Base Instructions
            (0b0000000, 0b000) => rs1_val.wrapping_add(rs2_val),                    // ADD
            (0b0100000, 0b000) => rs1_val.wrapping_sub(rs2_val),                    // SUB
            (0b0000000, 0b001) => rs1_val << (rs2_val & 0x3f),                      // SLL
            (0b0000000, 0b010) => if (rs1_val as i64) < (rs2_val as i64) { 1 } else { 0 }, // SLT
            (0b0000000, 0b011) => if rs1_val < rs2_val { 1 } else { 0 },            // SLTU
            (0b0000000, 0b100) => rs1_val ^ rs2_val,                                // XOR
            (0b0000000, 0b101) => rs1_val >> (rs2_val & 0x3f),                      // SRL
            (0b0100000, 0b101) => ((rs1_val as i64) >> (rs2_val & 0x3f)) as u64,   // SRA
            (0b0000000, 0b110) => rs1_val | rs2_val,                                // OR
            (0b0000000, 0b111) => rs1_val & rs2_val,                                // AND

            // RV32M Extension Instructions (Multiplication and Division)
            (0b0000001, 0b000) => {
                // MUL: Multiply (lower 32/64 bits)
                let result = (rs1_val as u128).wrapping_mul(rs2_val as u128) as u64;
                if self.debug_mode {
                    println!("MUL: {} * {} = {} (lower bits)", rs1_val, rs2_val, result);
                }
                result
            },
            (0b0000001, 0b001) => {
                // MULH: Multiply high (signed x signed)
                let result = ((rs1_val as i64 as i128) * (rs2_val as i64 as i128)) >> 64;
                if self.debug_mode {
                    println!("MULH: {} * {} = {} (upper bits, signed)", 
                             rs1_val as i64, rs2_val as i64, result);
                }
                result as u64
            },
            (0b0000001, 0b010) => {
                // MULHSU: Multiply high (signed x unsigned)
                let result = ((rs1_val as i64 as i128) * (rs2_val as u128 as i128)) >> 64;
                if self.debug_mode {
                    println!("MULHSU: {} * {} = {} (upper bits, mixed)", 
                             rs1_val as i64, rs2_val, result);
                }
                result as u64
            },
            (0b0000001, 0b011) => {
                // MULHU: Multiply high (unsigned x unsigned)
                let result = ((rs1_val as u128) * (rs2_val as u128)) >> 64;
                if self.debug_mode {
                    println!("MULHU: {} * {} = {} (upper bits, unsigned)", 
                             rs1_val, rs2_val, result);
                }
                result as u64
            },
            (0b0000001, 0b100) => {
                // DIV: Divide (signed)
                if rs2_val == 0 {
                    // Division by zero results in -1 for signed division
                    0xffffffffffffffff
                } else if rs1_val == 0x8000000000000000 && rs2_val as i64 == -1 {
                    // Overflow case: most negative / -1
                    0x8000000000000000
                } else {
                    let result = (rs1_val as i64) / (rs2_val as i64);
                    if self.debug_mode {
                        println!("DIV: {} / {} = {}", rs1_val as i64, rs2_val as i64, result);
                    }
                    result as u64
                }
            },
            (0b0000001, 0b101) => {
                // DIVU: Divide (unsigned)
                if rs2_val == 0 {
                    // Division by zero results in all 1s for unsigned division
                    0xffffffffffffffff
                } else {
                    let result = rs1_val / rs2_val;
                    if self.debug_mode {
                        println!("DIVU: {} / {} = {}", rs1_val, rs2_val, result);
                    }
                    result
                }
            },
            (0b0000001, 0b110) => {
                // REM: Remainder (signed)
                if rs2_val == 0 {
                    // Remainder by zero returns dividend
                    rs1_val
                } else if rs1_val == 0x8000000000000000 && rs2_val as i64 == -1 {
                    // Overflow case: remainder is 0
                    0
                } else {
                    let result = (rs1_val as i64) % (rs2_val as i64);
                    if self.debug_mode {
                        println!("REM: {} % {} = {}", rs1_val as i64, rs2_val as i64, result);
                    }
                    result as u64
                }
            },
            (0b0000001, 0b111) => {
                // REMU: Remainder (unsigned)
                if rs2_val == 0 {
                    // Remainder by zero returns dividend
                    rs1_val
                } else {
                    let result = rs1_val % rs2_val;
                    if self.debug_mode {
                        println!("REMU: {} % {} = {}", rs1_val, rs2_val, result);
                    }
                    result
                }
            },
            _ => return Err(format!("Invalid R-type: funct7={:07b}, funct3={:03b}", funct7, funct3)),
        };

        self.write_register(rd, result);
        self.pc += 4;
        Ok(())
    }

    fn execute_load(&mut self, rd: usize, funct3: u32, rs1: usize, inst: u32) -> Result<(), String> {
        let imm = ((inst as i32) >> 20) as i64;
        let addr = (self.read_register(rs1) as i64).wrapping_add(imm) as u64;

        let value = match funct3 {
            0b000 => {
                // LB: Load byte (sign-extended)
                let byte_val = self.read_memory(addr, 1) as u8;
                (byte_val as i8) as i64 as u64
            },
            0b001 => {
                // LH: Load halfword (sign-extended)
                let half_val = self.read_memory(addr, 2) as u16;
                (half_val as i16) as i64 as u64
            },
            0b010 => {
                // LW: Load word (sign-extended)
                let word_val = self.read_memory(addr, 4) as u32;
                (word_val as i32) as i64 as u64
            },
            0b011 => {
                // LD: Load doubleword (RV64 only)
                self.read_memory(addr, 8)
            },
            0b100 => {
                // LBU: Load byte unsigned
                self.read_memory(addr, 1) & 0xff
            },
            0b101 => {
                // LHU: Load halfword unsigned
                self.read_memory(addr, 2) & 0xffff
            },
            0b110 => {
                // LWU: Load word unsigned (RV64 only)
                self.read_memory(addr, 4) & 0xffffffff
            },
            _ => return Err(format!("Invalid load funct3: {}", funct3)),
        };

        if self.debug_mode {
            println!("Load: addr=0x{:08x}, value=0x{:016x}, type={}", 
                     addr, value, match funct3 {
                0b000 => "LB", 0b001 => "LH", 0b010 => "LW", 0b011 => "LD",
                0b100 => "LBU", 0b101 => "LHU", 0b110 => "LWU", _ => "?",
            });
        }

        self.write_register(rd, value);
        self.pc += 4;
        Ok(())
    }

    fn execute_store(&mut self, funct3: u32, rs1: usize, rs2: usize, inst: u32) -> Result<(), String> {
        // Decode S-type immediate
        let imm_11_5 = (inst >> 25) & 0x7f;
        let imm_4_0 = (inst >> 7) & 0x1f;
        let imm = ((imm_11_5 << 5) | imm_4_0) as i32;
        let imm = if imm & 0x800 != 0 { imm | !0xfff } else { imm };

        let addr = (self.read_register(rs1) as i64).wrapping_add(imm as i64) as u64;
        let value = self.read_register(rs2);

        match funct3 {
            0b000 => {
                // SB: Store byte
                self.write_memory(addr, value & 0xff, 1);
                if self.debug_mode {
                    println!("SB: addr=0x{:08x}, value=0x{:02x}", addr, value & 0xff);
                }
            },
            0b001 => {
                // SH: Store halfword
                self.write_memory(addr, value & 0xffff, 2);
                if self.debug_mode {
                    println!("SH: addr=0x{:08x}, value=0x{:04x}", addr, value & 0xffff);
                }
            },
            0b010 => {
                // SW: Store word
                self.write_memory(addr, value & 0xffffffff, 4);
                if self.debug_mode {
                    println!("SW: addr=0x{:08x}, value=0x{:08x}", addr, value & 0xffffffff);
                }
            },
            0b011 => {
                // SD: Store doubleword (RV64 only)
                self.write_memory(addr, value, 8);
                if self.debug_mode {
                    println!("SD: addr=0x{:08x}, value=0x{:016x}", addr, value);
                }
            },
            _ => return Err(format!("Invalid store funct3: {}", funct3)),
        }

        self.pc += 4;
        Ok(())
    }

    fn execute_branch(&mut self, funct3: u32, rs1: usize, rs2: usize, inst: u32) -> Result<(), String> {
        // Decode B-type immediate
        let imm_12 = (inst >> 31) & 0x1;
        let imm_10_5 = (inst >> 25) & 0x3f;
        let imm_4_1 = (inst >> 8) & 0xf;
        let imm_11 = (inst >> 7) & 0x1;
        let imm = (imm_12 << 12) | (imm_11 << 11) | (imm_10_5 << 5) | (imm_4_1 << 1);
        let imm = if imm & 0x1000 != 0 { (imm | !0x1fff) as i32 } else { imm as i32 };

        let rs1_val = self.read_register(rs1);
        let rs2_val = self.read_register(rs2);

        let take_branch = match funct3 {
            0b000 => {
                // BEQ: Branch if equal
                let result = rs1_val == rs2_val;
                if self.debug_mode {
                    println!("BEQ: {} == {} ? {}", rs1_val, rs2_val, result);
                }
                result
            },
            0b001 => {
                // BNE: Branch if not equal
                let result = rs1_val != rs2_val;
                if self.debug_mode {
                    println!("BNE: {} != {} ? {}", rs1_val, rs2_val, result);
                }
                result
            },
            0b100 => {
                // BLT: Branch if less than (signed)
                let result = (rs1_val as i64) < (rs2_val as i64);
                if self.debug_mode {
                    println!("BLT: {} < {} ? {}", rs1_val as i64, rs2_val as i64, result);
                }
                result
            },
            0b101 => {
                // BGE: Branch if greater or equal (signed)
                let result = (rs1_val as i64) >= (rs2_val as i64);
                if self.debug_mode {
                    println!("BGE: {} >= {} ? {}", rs1_val as i64, rs2_val as i64, result);
                }
                result
            },
            0b110 => {
                // BLTU: Branch if less than (unsigned)
                let result = rs1_val < rs2_val;
                if self.debug_mode {
                    println!("BLTU: {} < {} ? {}", rs1_val, rs2_val, result);
                }
                result
            },
            0b111 => {
                // BGEU: Branch if greater or equal (unsigned)
                let result = rs1_val >= rs2_val;
                if self.debug_mode {
                    println!("BGEU: {} >= {} ? {}", rs1_val, rs2_val, result);
                }
                result
            },
            _ => return Err(format!("Invalid branch funct3: {}", funct3)),
        };

        if take_branch {
            let new_pc = (self.pc as i64).wrapping_add(imm as i64) as u64;
            if self.debug_mode {
                println!("Branch taken: PC 0x{:08x} -> 0x{:08x} (offset={})", 
                         self.pc, new_pc, imm);
            }
            self.pc = new_pc;
        } else {
            if self.debug_mode {
                println!("Branch not taken: PC 0x{:08x} -> 0x{:08x}", self.pc, self.pc + 4);
            }
            self.pc += 4;
        }
        Ok(())
    }

    fn execute_jal(&mut self, rd: usize, inst: u32) -> Result<(), String> {
        // Decode J-type immediate
        let imm_20 = (inst >> 31) & 0x1;
        let imm_10_1 = (inst >> 21) & 0x3ff;
        let imm_11 = (inst >> 20) & 0x1;
        let imm_19_12 = (inst >> 12) & 0xff;
        let imm = (imm_20 << 20) | (imm_19_12 << 12) | (imm_11 << 11) | (imm_10_1 << 1);
        let imm = if imm & 0x100000 != 0 { (imm | !0x1fffff) as i32 } else { imm as i32 };

        let return_addr = self.pc + 4;
        let target_addr = (self.pc as i64).wrapping_add(imm as i64) as u64;

        if self.debug_mode {
            println!("JAL: return_addr=0x{:08x}, target=0x{:08x}, offset={}", 
                     return_addr, target_addr, imm);
        }

        self.write_register(rd, return_addr);
        self.pc = target_addr;
        Ok(())
    }

    fn execute_jalr(&mut self, rd: usize, rs1: usize, inst: u32) -> Result<(), String> {
        let imm = ((inst as i32) >> 20) as i64;
        let target = ((self.read_register(rs1) as i64).wrapping_add(imm) & !1) as u64;
        let return_addr = self.pc + 4;

        if self.debug_mode {
            println!("JALR: return_addr=0x{:08x}, target=0x{:08x}, base=0x{:08x}, offset={}", 
                     return_addr, target, self.read_register(rs1), imm);
        }

        self.write_register(rd, return_addr);
        self.pc = target;
        Ok(())
    }

    fn execute_lui(&mut self, rd: usize, inst: u32) -> Result<(), String> {
        let imm = (inst & 0xfffff000) as i32;
        let value = imm as i64 as u64;
        
        if self.debug_mode {
            println!("LUI: rd=x{}, imm=0x{:08x}, value=0x{:016x}", rd, imm, value);
        }
        
        self.write_register(rd, value);
        self.pc += 4;
        Ok(())
    }

    fn execute_auipc(&mut self, rd: usize, inst: u32) -> Result<(), String> {
        let imm = (inst & 0xfffff000) as i32;
        let result = (self.pc as i64).wrapping_add(imm as i64) as u64;
        
        if self.debug_mode {
            println!("AUIPC: rd=x{}, pc=0x{:08x}, imm=0x{:08x}, result=0x{:016x}", 
                     rd, self.pc, imm, result);
        }
        
        self.write_register(rd, result);
        self.pc += 4;
        Ok(())
    }

    fn execute_system(&mut self, inst: u32) -> Result<(), String> {
        let funct3 = (inst >> 12) & 0x07;
        let imm = inst >> 20;

        match (funct3, imm) {
            (0b000, 0b000000000000) => {
                // ECALL: Environment call
                if self.debug_mode {
                    println!("ECALL: System call requested");
                }
                // In a real implementation, this would trap to the OS
                self.pc += 4;
                Ok(())
            },
            (0b000, 0b000000000001) => {
                // EBREAK: Environment break
                if self.debug_mode {
                    println!("EBREAK: Breakpoint encountered");
                }
                // In a real implementation, this would trap to the debugger
                self.pc += 4;
                Ok(())
            },
            _ => {
                if self.debug_mode {
                    println!("System instruction: funct3={}, imm=0x{:03x}", funct3, imm);
                }
                self.pc += 4;
                Ok(())
            }
        }
    }

    fn execute_fence(&mut self, _inst: u32) -> Result<(), String> {
        // FENCE instructions for memory ordering
        // In a simple implementation, we can treat these as NOPs
        if self.debug_mode {
            println!("FENCE: Memory ordering instruction");
        }
        self.pc += 4;
        Ok(())
    }

    pub fn disassemble_instruction(&self, inst: u32) -> String {
        let opcode = inst & 0x7f;
        let rd = (inst >> 7) & 0x1f;
        let funct3 = (inst >> 12) & 0x07;
        let rs1 = (inst >> 15) & 0x1f;
        let rs2 = (inst >> 20) & 0x1f;
        let funct7 = (inst >> 25) & 0x7f;

        match opcode {
            0b0010011 => {
                let imm = (inst as i32) >> 20;
                match funct3 {
                    0b000 => format!("addi x{}, x{}, {}", rd, rs1, imm),
                    0b010 => format!("slti x{}, x{}, {}", rd, rs1, imm),
                    0b011 => format!("sltiu x{}, x{}, {}", rd, rs1, imm),
                    0b100 => format!("xori x{}, x{}, {}", rd, rs1, imm),
                    0b110 => format!("ori x{}, x{}, {}", rd, rs1, imm),
                    0b111 => format!("andi x{}, x{}, {}", rd, rs1, imm),
                    0b001 => format!("slli x{}, x{}, {}", rd, rs1, imm & 0x3f),
                    0b101 => {
                        if (inst >> 30) & 1 == 0 {
                            format!("srli x{}, x{}, {}", rd, rs1, imm & 0x3f)
                        } else {
                            format!("srai x{}, x{}, {}", rd, rs1, imm & 0x3f)
                        }
                    },
                    _ => format!("unknown_op_imm 0x{:08x}", inst),
                }
            },
            0b0110011 => {
                match (funct7, funct3) {
                    (0b0000000, 0b000) => format!("add x{}, x{}, x{}", rd, rs1, rs2),
                    (0b0100000, 0b000) => format!("sub x{}, x{}, x{}", rd, rs1, rs2),
                    (0b0000000, 0b001) => format!("sll x{}, x{}, x{}", rd, rs1, rs2),
                    (0b0000000, 0b010) => format!("slt x{}, x{}, x{}", rd, rs1, rs2),
                    (0b0000000, 0b011) => format!("sltu x{}, x{}, x{}", rd, rs1, rs2),
                    (0b0000000, 0b100) => format!("xor x{}, x{}, x{}", rd, rs1, rs2),
                    (0b0000000, 0b101) => format!("srl x{}, x{}, x{}", rd, rs1, rs2),
                    (0b0100000, 0b101) => format!("sra x{}, x{}, x{}", rd, rs1, rs2),
                    (0b0000000, 0b110) => format!("or x{}, x{}, x{}", rd, rs1, rs2),
                    (0b0000000, 0b111) => format!("and x{}, x{}, x{}", rd, rs1, rs2),
                    // M Extension
                    (0b0000001, 0b000) => format!("mul x{}, x{}, x{}", rd, rs1, rs2),
                    (0b0000001, 0b001) => format!("mulh x{}, x{}, x{}", rd, rs1, rs2),
                    (0b0000001, 0b010) => format!("mulhsu x{}, x{}, x{}", rd, rs1, rs2),
                    (0b0000001, 0b011) => format!("mulhu x{}, x{}, x{}", rd, rs1, rs2),
                    (0b0000001, 0b100) => format!("div x{}, x{}, x{}", rd, rs1, rs2),
                    (0b0000001, 0b101) => format!("divu x{}, x{}, x{}", rd, rs1, rs2),
                    (0b0000001, 0b110) => format!("rem x{}, x{}, x{}", rd, rs1, rs2),
                    (0b0000001, 0b111) => format!("remu x{}, x{}, x{}", rd, rs1, rs2),
                    _ => format!("unknown_op 0x{:08x}", inst),
                }
            },
            0b0000011 => {
                let imm = (inst as i32) >> 20;
                match funct3 {
                    0b000 => format!("lb x{}, {}(x{})", rd, imm, rs1),
                    0b001 => format!("lh x{}, {}(x{})", rd, imm, rs1),
                    0b010 => format!("lw x{}, {}(x{})", rd, imm, rs1),
                    0b011 => format!("ld x{}, {}(x{})", rd, imm, rs1),
                    0b100 => format!("lbu x{}, {}(x{})", rd, imm, rs1),
                    0b101 => format!("lhu x{}, {}(x{})", rd, imm, rs1),
                    0b110 => format!("lwu x{}, {}(x{})", rd, imm, rs1),
                    _ => format!("unknown_load 0x{:08x}", inst),
                }
            },
            0b0100011 => {
                let imm_11_5 = (inst >> 25) & 0x7f;
                let imm_4_0 = (inst >> 7) & 0x1f;
                let imm = ((imm_11_5 << 5) | imm_4_0) as i32;
                let imm = if imm & 0x800 != 0 { imm | !0xfff } else { imm };
                
                match funct3 {
                    0b000 => format!("sb x{}, {}(x{})", rs2, imm, rs1),
                    0b001 => format!("sh x{}, {}(x{})", rs2, imm, rs1),
                    0b010 => format!("sw x{}, {}(x{})", rs2, imm, rs1),
                    0b011 => format!("sd x{}, {}(x{})", rs2, imm, rs1),
                    _ => format!("unknown_store 0x{:08x}", inst),
                }
            },
            0b1100011 => {
                let imm_12 = (inst >> 31) & 0x1;
                let imm_10_5 = (inst >> 25) & 0x3f;
                let imm_4_1 = (inst >> 8) & 0xf;
                let imm_11 = (inst >> 7) & 0x1;
                let imm = (imm_12 << 12) | (imm_11 << 11) | (imm_10_5 << 5) | (imm_4_1 << 1);
                let imm = if imm & 0x1000 != 0 { (imm | !0x1fff) as i32 } else { imm as i32 };
                
                match funct3 {
                    0b000 => format!("beq x{}, x{}, {}", rs1, rs2, imm),
                    0b001 => format!("bne x{}, x{}, {}", rs1, rs2, imm),
                    0b100 => format!("blt x{}, x{}, {}", rs1, rs2, imm),
                    0b101 => format!("bge x{}, x{}, {}", rs1, rs2, imm),
                    0b110 => format!("bltu x{}, x{}, {}", rs1, rs2, imm),
                    0b111 => format!("bgeu x{}, x{}, {}", rs1, rs2, imm),
                    _ => format!("unknown_branch 0x{:08x}", inst),
                }
            },
            0b1101111 => {
                let imm_20 = (inst >> 31) & 0x1;
                let imm_10_1 = (inst >> 21) & 0x3ff;
                let imm_11 = (inst >> 20) & 0x1;
                let imm_19_12 = (inst >> 12) & 0xff;
                let imm = (imm_20 << 20) | (imm_19_12 << 12) | (imm_11 << 11) | (imm_10_1 << 1);
                let imm = if imm & 0x100000 != 0 { (imm | !0x1fffff) as i32 } else { imm as i32 };
                
                format!("jal x{}, {}", rd, imm)
            },
            0b1100111 => {
                let imm = (inst as i32) >> 20;
                format!("jalr x{}, {}(x{})", rd, imm, rs1)
            },
            0b0110111 => {
                let imm = (inst & 0xfffff000) as i32;
                format!("lui x{}, 0x{:x}", rd, (imm as u32) >> 12)
            },
            0b0010111 => {
                let imm = (inst & 0xfffff000) as i32;
                format!("auipc x{}, 0x{:x}", rd, (imm as u32) >> 12)
            },
            _ => format!("unknown 0x{:08x}", inst),
        }
    }

    pub fn run(&mut self, max_steps: usize) -> Result<Vec<ExecutionTrace>, String> {
        let mut traces = Vec::new();
        
        for step in 0..max_steps {
            if self.debug_mode {
                println!("\n=== Step {} ===", step);
                println!("PC: 0x{:08x}", self.pc);
            }
            
            let trace = self.step()?;
            traces.push(trace);
            
            if self.debug_mode {
                self.print_registers();
            }
        }
        
        Ok(traces)
    }

    pub fn print_registers(&self) {
        println!("Registers:");
        for i in 0..32 {
            if i % 4 == 0 { print!("\n"); }
            print!("x{:2}=0x{:016x} ", i, self.read_register(i));
        }
        println!("\nPC=0x{:016x}", self.pc);
        println!("Instructions: {}, Cycles: {}", self.instruction_count, self.cycle_count);
    }
}

// Test programs demonstrating various features
pub fn all() {
    println!("=== RISC-V Extended CPU Test ===\n");

    // Test 1: Multiplication and Division
    test_multiplication_division();
    
    // Test 2: Branching and Loops
    test_branching_loops();
    
    // Test 3: Memory Operations
    test_memory_operations();
    
    // Test 4: Fibonacci Sequence
    test_fibonacci();
}

fn test_multiplication_division() {
    println!("=== Test 1: Multiplication and Division ===");
    let mut cpu = ExtendedCpu::new();
    cpu.enable_debug();

    // Test program: multiplication and division
    let program = vec![
        0x93, 0x00, 0xc0, 0x00,  // addi x1, x0, 12      (x1 = 12)
        0x13, 0x01, 0x50, 0x00,  // addi x2, x0, 5       (x2 = 5)
        0xb3, 0x81, 0x20, 0x02,  // mul x3, x1, x2       (x3 = 12 * 5 = 60)
        0x33, 0xc2, 0x20, 0x02,  // div x4, x1, x2       (x4 = 12 / 5 = 2)
        0xb3, 0xe2, 0x20, 0x02,  // rem x5, x1, x2       (x5 = 12 % 5 = 2)
    ];

    cpu.load_program(&program, 0x1000);
    
    if let Err(e) = cpu.run(5) {
        println!("Error: {}", e);
    }

    println!("Expected: x1=12, x2=5, x3=60, x4=2, x5=2");
    println!("Actual: x1={}, x2={}, x3={}, x4={}, x5={}", 
             cpu.read_register(1), cpu.read_register(2), cpu.read_register(3),
             cpu.read_register(4), cpu.read_register(5));
    println!();
}

fn test_branching_loops() {
    println!("=== Test 2: Branching and Loops ===");
    let mut cpu = ExtendedCpu::new();
    cpu.enable_debug();

    // Simple loop: count from 0 to 3
    let program = vec![
        0x93, 0x00, 0x00, 0x00,  // addi x1, x0, 0       (counter = 0)
        0x13, 0x01, 0x30, 0x00,  // addi x2, x0, 3       (limit = 3)
        // loop:
        0x93, 0x80, 0x10, 0x00,  // addi x1, x1, 1       (counter++)
        0x63, 0xd0, 0x20, 0x00,  // bge x1, x2, 8        (if counter >= limit, exit)
        0x6f, 0x00, 0x80, 0xff,  // jal x0, -8           (jump back to loop)
        // exit:
        0x13, 0x81, 0xf0, 0xff,  // addi x2, x1, -1      (final result)
    ];

    cpu.load_program(&program, 0x1000);
    
    if let Err(e) = cpu.run(15) {
        println!("Error: {}", e);
    }

    println!("Expected: x1=3 (final counter), x2=2 (counter-1)");
    println!("Actual: x1={}, x2={}", cpu.read_register(1), cpu.read_register(2));
    println!();
}

fn test_memory_operations() {
    println!("=== Test 3: Memory Operations ===");
    let mut cpu = ExtendedCpu::new();
    cpu.enable_debug();

    // Test different store/load sizes
    let program = vec![
        0x37, 0x12, 0x34, 0x56,  // lui x4, 0x56341     (x4 = 0x56341000)
        0x13, 0x02, 0x82, 0x23,  // addi x4, x4, 0x238  (x4 = 0x56341238)
        0x37, 0x15, 0x00, 0x20,  // lui x10, 0x20001    (base address)
        
        // Store operations
        0x23, 0x00, 0x45, 0x00,  // sb x4, 0(x10)       (store byte)
        0x23, 0x11, 0x45, 0x00,  // sh x4, 2(x10)       (store halfword)
        0x23, 0x22, 0x45, 0x00,  // sw x4, 4(x10)       (store word)
        
        // Load operations
        0x03, 0x05, 0x05, 0x00,  // lb x10, 0(x10)      (load byte signed)
        0x83, 0x45, 0x25, 0x00,  // lbu x11, 2(x10)     (load byte unsigned)
        0x03, 0x16, 0x45, 0x00,  // lh x12, 4(x10)      (load halfword)
    ];

    cpu.load_program(&program, 0x1000);
    
    if let Err(e) = cpu.run(9) {
        println!("Error: {}", e);
    }

    println!("Memory operations completed - check debug output for details");
    println!();
}

fn test_fibonacci() {
    println!("=== Test 4: Fibonacci Sequence (First 6 Numbers) ===");
    let mut cpu = ExtendedCpu::new();
    cpu.enable_debug();

    // Calculate fibonacci sequence: 1, 1, 2, 3, 5, 8
    let program = vec![
        0x93, 0x00, 0x10, 0x00,  // addi x1, x0, 1       (fib_a = 1)
        0x13, 0x01, 0x10, 0x00,  // addi x2, x0, 1       (fib_b = 1)
        0x93, 0x01, 0x60, 0x00,  // addi x3, x0, 6       (count = 6)
        0x13, 0x02, 0x00, 0x00,  // addi x4, x0, 0       (i = 0)
        
        // fib_loop:
        0x33, 0x82, 0x20, 0x00,  // add x5, x1, x2       (temp = fib_a + fib_b)
        0x93, 0x80, 0x20, 0x00,  // add x1, x2, x0       (fib_a = fib_b)
        0x13, 0x01, 0x50, 0x00,  // add x2, x5, x0       (fib_b = temp)
        0x13, 0x82, 0x12, 0x00,  // addi x4, x4, 1       (i++)
        0x63, 0xc8, 0x32, 0x00,  // blt x4, x3, -16      (if i < count, loop)
        
        // Result in x2 should be 8th fibonacci number
    ];

    cpu.load_program(&program, 0x1000);
    
    if let Err(e) = cpu.run(30) {
        println!("Error: {}", e);
    }

    println!("Expected final result: x2=13 (6th fibonacci number)");
    println!("Actual: x1={}, x2={}, count={}, i={}", 
             cpu.read_register(1), cpu.read_register(2), 
             cpu.read_register(3), cpu.read_register(4));
}