use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct Cpu {
    pub registers: [u64; 32],
    pub pc: u64,
    pub memory: HashMap<u64, u8>,
    pub execution_trace: Vec<ExecutionStep>,
}

#[derive(Debug, Clone)]
pub struct ExecutionStep {
    pub pc: u64,
    pub instruction: u32,
    pub opcode: u8,
    pub rd: u8,
    pub rs1: u8,
    pub rs2: u8,
    pub imm: i32,
    pub funct3: u8,
    pub funct7: u8,
    pub registers_before: [u64; 32],
    pub registers_after: [u64; 32],
}

impl Cpu {
    pub fn new() -> Self {
        Self {
            registers: [0; 32],
            pc: 0,
            memory: HashMap::new(),
            execution_trace: Vec::new(),
        }
    }

    pub fn load_program(&mut self, program: &[u8], start_addr: u64) {
        for (i, &byte) in program.iter().enumerate() {
            self.memory.insert(start_addr + i as u64, byte);
        }
        self.pc = start_addr;
    }

    pub fn read_memory(&self, addr: u64, bytes: usize) -> u64 {
        let mut result = 0u64;
        for i in 0..bytes {
            if let Some(&byte) = self.memory.get(&(addr + i as u64)) {
                result |= (byte as u64) << (i * 8);
            }
        }
        result
    }

    pub fn write_memory(&mut self, addr: u64, value: u64, bytes: usize) {
        for i in 0..bytes {
            let byte = ((value >> (i * 8)) & 0xff) as u8;
            self.memory.insert(addr + i as u64, byte);
        }
    }

    pub fn read_register(&self, reg: usize) -> u64 {
        if reg == 0 { 0 } else { self.registers[reg] }
    }

    pub fn write_register(&mut self, reg: usize, value: u64) {
        if reg != 0 {
            self.registers[reg] = value;
        }
    }

    pub fn fetch_instruction(&self) -> u32 {
        self.read_memory(self.pc, 4) as u32
    }

    pub fn step(&mut self) -> Result<(), String> {
        let inst = self.fetch_instruction();
        let registers_before = self.registers.clone();
        
        self.execute_instruction(inst)?;
        
        // Record execution step for winterfell
        let registers_after = self.registers.clone();
        self.record_execution_step(inst, registers_before, registers_after);
        
        Ok(())
    }

    fn record_execution_step(&mut self, inst: u32, registers_before: [u64; 32], registers_after: [u64; 32]) {
        let opcode = (inst & 0x7f) as u8;
        let rd = ((inst >> 7) & 0x1f) as u8;
        let rs1 = ((inst >> 15) & 0x1f) as u8;
        let rs2 = ((inst >> 20) & 0x1f) as u8;
        let funct3 = ((inst >> 12) & 0x07) as u8;
        let funct7 = ((inst >> 25) & 0x7f) as u8;
        
        // Extract immediate based on instruction type
        let imm = match opcode {
            0b0010011 => (inst as i32) >> 20, // I-type
            0b0110011 => 0, // R-type
            _ => 0,
        };

        let step = ExecutionStep {
            pc: self.pc - 4,
            instruction: inst,
            opcode,
            rd,
            rs1,
            rs2,
            imm,
            funct3,
            funct7,
            registers_before,
            registers_after,
        };
        
        self.execution_trace.push(step);
    }

    pub fn execute_instruction(&mut self, inst: u32) -> Result<(), String> {
        let opcode = inst & 0x7f;
        let rd = ((inst >> 7) & 0x1f) as usize;
        let funct3 = (inst >> 12) & 0x07;
        let rs1 = ((inst >> 15) & 0x1f) as usize;
        let rs2 = ((inst >> 20) & 0x1f) as usize;
        let funct7 = (inst >> 25) & 0x7f;

        println!("Instruction: 0x{:08x}", inst);
        println!("  Opcode: 0b{:07b} (0x{:02x})", opcode, opcode);
        println!("  rd: {}, rs1: {}, rs2: {}", rd, rs1, rs2);
        println!("  funct3: 0b{:03b}, funct7: 0b{:07b}", funct3, funct7);

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
            _ => {
                println!("Unknown opcode: 0b{:07b} (0x{:08x})", opcode, inst);
                self.pc += 4;
                Ok(())
            }
        }
    }

    fn execute_op_imm(&mut self, inst: u32, rd: usize, funct3: u32, rs1: usize) -> Result<(), String> {
        let imm = ((inst as i32) >> 20) as i64 as u64;
        let rs1_val = self.read_register(rs1);

        println!("  OP-IMM: rs1_val={}, imm={} (0x{:x})", rs1_val, imm as i64, imm);

        let result = match funct3 {
            0b000 => rs1_val.wrapping_add(imm), // ADDI
            0b010 => if (rs1_val as i64) < (imm as i64) { 1 } else { 0 }, // SLTI
            0b011 => if rs1_val < imm { 1 } else { 0 }, // SLTIU
            0b100 => rs1_val ^ imm, // XORI
            0b110 => rs1_val | imm, // ORI
            0b111 => rs1_val & imm, // ANDI
            0b001 => { // SLLI
                let shamt = (imm & 0x3f) as u32;
                rs1_val << shamt
            },
            0b101 => { // SRLI/SRAI
                let shamt = (imm & 0x3f) as u32;
                if (inst >> 30) & 1 == 0 {
                    rs1_val >> shamt // SRLI
                } else {
                    ((rs1_val as i64) >> shamt) as u64 // SRAI
                }
            },
            _ => return Err(format!("Invalid funct3 for OP-IMM: {}", funct3)),
        };

        println!("  Result: {}", result);
        self.write_register(rd, result);
        self.pc += 4;
        Ok(())
    }

    fn execute_op(&mut self, rd: usize, funct3: u32, rs1: usize, rs2: usize, funct7: u32) -> Result<(), String> {
        let rs1_val = self.read_register(rs1);
        let rs2_val = self.read_register(rs2);

        println!("  OP: rs1_val={}, rs2_val={}", rs1_val, rs2_val);

        let result = match (funct7, funct3) {
            (0b0000000, 0b000) => rs1_val.wrapping_add(rs2_val), // ADD
            (0b0100000, 0b000) => rs1_val.wrapping_sub(rs2_val), // SUB
            (0b0000000, 0b001) => rs1_val << (rs2_val & 0x3f), // SLL
            (0b0000000, 0b010) => if (rs1_val as i64) < (rs2_val as i64) { 1 } else { 0 }, // SLT
            (0b0000000, 0b011) => if rs1_val < rs2_val { 1 } else { 0 }, // SLTU
            (0b0000000, 0b100) => rs1_val ^ rs2_val, // XOR
            (0b0000000, 0b101) => rs1_val >> (rs2_val & 0x3f), // SRL
            (0b0100000, 0b101) => ((rs1_val as i64) >> (rs2_val & 0x3f)) as u64, // SRA
            (0b0000000, 0b110) => rs1_val | rs2_val, // OR
            (0b0000000, 0b111) => rs1_val & rs2_val, // AND
            _ => return Err(format!("Invalid R-type: funct7={:07b}, funct3={:03b}", funct7, funct3)),
        };

        println!("  Result: {}", result);
        self.write_register(rd, result);
        self.pc += 4;
        Ok(())
    }

    fn execute_load(&mut self, rd: usize, funct3: u32, rs1: usize, inst: u32) -> Result<(), String> {
        let imm = ((inst as i32) >> 20) as i64;
        let addr = (self.read_register(rs1) as i64).wrapping_add(imm) as u64;

        let value = match funct3 {
            0b000 => (self.read_memory(addr, 1) as i8) as i64 as u64,
            0b001 => (self.read_memory(addr, 2) as i16) as i64 as u64,
            0b010 => (self.read_memory(addr, 4) as i32) as i64 as u64,
            0b011 => self.read_memory(addr, 8),
            0b100 => self.read_memory(addr, 1),
            0b101 => self.read_memory(addr, 2),
            0b110 => self.read_memory(addr, 4),
            _ => return Err(format!("Invalid load funct3: {}", funct3)),
        };

        self.write_register(rd, value);
        self.pc += 4;
        Ok(())
    }

    fn execute_store(&mut self, funct3: u32, rs1: usize, rs2: usize, inst: u32) -> Result<(), String> {
        let imm_11_5 = (inst >> 25) & 0x7f;
        let imm_4_0 = (inst >> 7) & 0x1f;
        let imm = ((imm_11_5 << 5) | imm_4_0) as i32;
        let imm = if imm & 0x800 != 0 { imm | !0xfff } else { imm };

        let addr = (self.read_register(rs1) as i64).wrapping_add(imm as i64) as u64;
        let value = self.read_register(rs2);

        match funct3 {
            0b000 => self.write_memory(addr, value, 1),
            0b001 => self.write_memory(addr, value, 2),
            0b010 => self.write_memory(addr, value, 4),
            0b011 => self.write_memory(addr, value, 8),
            _ => return Err(format!("Invalid store funct3: {}", funct3)),
        }

        self.pc += 4;
        Ok(())
    }

    fn execute_branch(&mut self, funct3: u32, rs1: usize, rs2: usize, inst: u32) -> Result<(), String> {
        let imm_12 = (inst >> 31) & 0x1;
        let imm_10_5 = (inst >> 25) & 0x3f;
        let imm_4_1 = (inst >> 8) & 0xf;
        let imm_11 = (inst >> 7) & 0x1;
        let imm = (imm_12 << 12) | (imm_11 << 11) | (imm_10_5 << 5) | (imm_4_1 << 1);
        let imm = if imm & 0x1000 != 0 { (imm | !0x1fff) as i32 } else { imm as i32 };

        let rs1_val = self.read_register(rs1);
        let rs2_val = self.read_register(rs2);

        let take_branch = match funct3 {
            0b000 => rs1_val == rs2_val,
            0b001 => rs1_val != rs2_val,
            0b100 => (rs1_val as i64) < (rs2_val as i64),
            0b101 => (rs1_val as i64) >= (rs2_val as i64),
            0b110 => rs1_val < rs2_val,
            0b111 => rs1_val >= rs2_val,
            _ => return Err(format!("Invalid branch funct3: {}", funct3)),
        };

        if take_branch {
            self.pc = (self.pc as i64).wrapping_add(imm as i64) as u64;
        } else {
            self.pc += 4;
        }
        Ok(())
    }

    fn execute_jal(&mut self, rd: usize, inst: u32) -> Result<(), String> {
        let imm_20 = (inst >> 31) & 0x1;
        let imm_10_1 = (inst >> 21) & 0x3ff;
        let imm_11 = (inst >> 20) & 0x1;
        let imm_19_12 = (inst >> 12) & 0xff;
        let imm = (imm_20 << 20) | (imm_19_12 << 12) | (imm_11 << 11) | (imm_10_1 << 1);
        let imm = if imm & 0x100000 != 0 { (imm | !0x1fffff) as i32 } else { imm as i32 };

        self.write_register(rd, self.pc + 4);
        self.pc = (self.pc as i64).wrapping_add(imm as i64) as u64;
        Ok(())
    }

    fn execute_jalr(&mut self, rd: usize, rs1: usize, inst: u32) -> Result<(), String> {
        let imm = ((inst as i32) >> 20) as i64;
        let target = ((self.read_register(rs1) as i64).wrapping_add(imm) & !1) as u64;

        self.write_register(rd, self.pc + 4);
        self.pc = target;
        Ok(())
    }

    fn execute_lui(&mut self, rd: usize, inst: u32) -> Result<(), String> {
        let imm = (inst & 0xfffff000) as i32;
        self.write_register(rd, imm as i64 as u64);
        self.pc += 4;
        Ok(())
    }

    fn execute_auipc(&mut self, rd: usize, inst: u32) -> Result<(), String> {
        let imm = (inst & 0xfffff000) as i32;
        let result = (self.pc as i64).wrapping_add(imm as i64) as u64;
        self.write_register(rd, result);
        self.pc += 4;
        Ok(())
    }

    pub fn run(&mut self, max_steps: usize) -> Result<(), String> {
        for step in 0..max_steps {
            println!("Step {}: PC=0x{:08x}", step, self.pc);
            self.print_registers();
            
            if let Err(e) = self.step() {
                return Err(format!("Error at step {}: {}", step, e));
            }
            
            println!();
        }
        Ok(())
    }

    pub fn print_registers(&self) {
        for i in 0..32 {
            if i % 4 == 0 { print!("\n"); }
            print!("x{:2}=0x{:016x} ", i, self.read_register(i));
        }
        println!();
    }

    pub fn get_execution_trace(&self) -> &Vec<ExecutionStep> {
        &self.execution_trace
    }

    pub fn export_trace_for_winterfell(&self) -> Vec<Vec<u64>> {
        // Convert execution trace to winterfell format
        let mut trace = Vec::new();
        
        for step in &self.execution_trace {
            let mut row = Vec::new();
            
            row.push(step.pc);
            row.push(step.instruction as u64);
            row.push(step.opcode as u64);
            row.push(step.rd as u64);
            row.push(step.rs1 as u64);
            row.push(step.rs2 as u64);
            row.push(step.imm as u64);
            
            for reg in &step.registers_before {
                row.push(*reg);
            }
            for reg in &step.registers_after {
                row.push(*reg);
            }
            
            trace.push(row);
        }
        
        trace
    }
}

fn main() {
    let mut cpu = Cpu::new();

    let program = vec![
        // addi x2, x0, 42    (x2 = 42)
        0x13, 0x01, 0xa0, 0x02,  
        // addi x3, x0, 21    (x3 = 21) 
        0x93, 0x01, 0x50, 0x01,  
        // add x4, x2, x3     (x4 = x2 + x3 = 63)
        0x33, 0x02, 0x31, 0x00,  
        // sub x5, x2, x3     (x5 = x2 - x3 = 21)
        0xb3, 0x02, 0x31, 0x40,  
    ];

    cpu.load_program(&program, 0x1000);
    
    println!("Initial state:");
    cpu.print_registers();
    println!("\nExecuting program...\n");

    if let Err(e) = cpu.run(4) {
        println!("Error: {}", e);
    }

    println!("Final state:");
    cpu.print_registers();
    
    println!("\nVerifying results:");
    println!("x2 should be 42: {}", cpu.read_register(2));
    println!("x3 should be 21: {}", cpu.read_register(3));
    println!("x4 should be 63: {}", cpu.read_register(4));
    println!("x5 should be 21: {}", cpu.read_register(5));

    // Export trace for winterfell
    let trace = cpu.export_trace_for_winterfell();
    println!("\nExecution trace exported: {} steps", trace.len());
}