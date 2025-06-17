import os
import re
import json
import csv
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--set_clock", action = "store_true", help="Set the clock to the model")
parser.add_argument("--set_while", action="store_true", help="Set the while loop in the model")
args = parser.parse_args()

if args.set_clock:
    mode = "clock"
elif args.set_while:
    mode = "while"
else:
    mode = "step"

#If/else/case format function
def get_if_else_statement_fmt(length: int, always_comb: bool = True, implicit_final_condition: bool = True, case_format: bool = False, unique: bool = False, default_assign: bool = True) -> str:
    """
    Generates a formatted string for an if-else or case statement in SystemVerilog.
    Args:
        length (int): The number of conditions to generate.
        always_comb (bool, optional): If True, wraps the statement in an always_comb block. Defaults to True.
        implicit_final_condition (bool, optional): If True, the final else condition is implicit. Defaults to True.  
        case_format (bool, optional): If True, generates a case statement instead of if-else. Defaults to False.
    Returns:
        str: A formatted string representing the if-else or case statement.
    """
    
    if always_comb:
        out_fmt = "always_comb begin\n"
    else:
        out_fmt = "\n"
        
    if case_format is True:
        
        if unique:
            out_fmt += "{indent}unique case ({val})\n"
        else:
            out_fmt += "{indent}casez ({val})\n\n"
        for i in range(length+1):
            out_fmt += f"{{indent}}{INDENT_ONE}{{condition{i}}} : begin\n\n{{indent}}{{indent}}{{assign{i}}}\n{{indent}}{INDENT_ONE}end\n\n"

        if default_assign is True:
            out_fmt += f"{{indent}}{INDENT_ONE}default: {{default_assign}}\n\n"    
        
        out_fmt += "{indent}endcase\n"
    else:
        for i in range(length):
            if i == 0:
                out_fmt += f"{{indent}}\tif ({{condition{i}}}) begin\n"
            elif i == length-1 and implicit_final_condition is True:
                out_fmt += "{indent}\tend else begin\n"
            else:
                out_fmt += f"{{indent}}\tend else if ({{condition{i}}}) begin\n"
                
            # out_fmt += f"{{indent}}\t\t{{lhs}} = {{rhs{i}}};\n"
            out_fmt += f"{{assign{i}}}"

        out_fmt += "{indent}\tend\n"
    
    if always_comb:
        out_fmt += "\tend\n"
    
    return out_fmt


#Function to reorder the casez_dict based on the priority list
def reorder_casez_dict(casez_dict: dict, priority_path: str) -> dict:

    priority_map = {name: i for i, name in enumerate(priority_list)}

    num_entries = len(casez_dict) // 2
    pair_list = []
    for i in range(num_entries):
        cond = casez_dict[f"condition{i}"]
        assign = casez_dict[f"assign{i}"]
        pair_list.append((cond, assign))

    sorted_pairs = sorted(pair_list, key=lambda x: priority_map.get(x[0], 1000))

    new_casez_dict = {}
    for i, (cond, assign) in enumerate(sorted_pairs):
        new_casez_dict[f"condition{i}"] = cond
        new_casez_dict[f"assign{i}"] = assign

    return new_casez_dict


def append_update(dict1, dict2):
    for key, value in dict2.items():
        if key in dict1:
            if isinstance(dict1[key], list):
                dict1[key].extend(value)
            else:
                if isinstance(value, dict):
                    # If the value is a dict, we need to update the dict1[key] recursively
                    append_update(dict1[key], value)
                else:
                    # For non-dict values, turn them into lists
                    dict1[key] = [dict1[key], value]
        else:
            dict1[key] = value


def read_json_files(json_files):
    instructions = {}
    for json_file, flag in json_files:
        # print(json_file, flag)
        if flag:
            with open(json_file, 'r') as f:
                data = json.load(f)
                append_update(instructions, data["instructions"])
    # print(instructions)
    return instructions



##################################################################################################################
#                                                                                                                #
#                                                   MAIN                                                         #
#                                                                                                                #
##################################################################################################################
config_json = 'config.json' 
instr_dict_json = 'instr_dict.json'
impl_dict_json = 'impl_dict.json'
input_file = 'inst.sverilog'
output_file = 'opcode_case_class.sv'
arg_lut_file = 'arg_lut.csv'
opcode_priority = "opcode_priority.json"


only_variable_fields = dict()  #Dictionary which will contain key = instruction's name and val = variable fields
opcode_dict = dict()           #Dictionary which will contain key = instruction's name and val = instruction's bit encoding
instruction_formats = dict()   #Dictionary which will contain key = instruction's name and val = instruction's type (R, I, S, ecc.)
bitfield_mapping = dict()      #Dictionary which will contain key = instruction's name and val = dictionary with the bitfield mapping for each variable fields
casez_dict = dict()            #Dictionary which will contain key = instruction's name and val = all the stuff to be put in the each case statement
implementations_dict = dict()  #Dictionary which will contain key = instruction's name and val = the implementation of the instruction

json_files = [
    ('isa_generic_ALU.json', True),
    ('isa_clip.json', True),
    ('isa_add_sub.json', True),
    ('isa_add_sub_ls3.json', True),
    ('isa_mac_32.json', True),
    ('isa_mul_16_8.json', True),
    ('isa_mac_16_8.json', True),
    ('isa_generic_SIMD.json', True),
    ('isa_dotp_SIMD.json', True),
    ('isa_cmp_SIMD.json', True),
]

# Read JSON files and collect instructions
dict_maurizio_instructions = read_json_files(json_files)

var_field_nuove_istruzioni = {}
for opcode in dict_maurizio_instructions:
    funct3_block = dict_maurizio_instructions[opcode]["funct3"]
    for funct3 in funct3_block:
        funct7_block = funct3_block[funct3]["funct7"]
        for funct7 in funct7_block:
            instr = funct7_block[funct7]
            mnemonic = instr["mnemonic"]
            operands = instr["operands"]
            fields = list(operands.keys())
            var_field_nuove_istruzioni[mnemonic] = fields  # Inserisci nel dizionario


#Global variables to make an indentation when needed
INDENT_ONE = "    "                     
INDENT_TWO = "        "
INDENT_THREE = "            "


# Define field dictionaries for different instruction types (7 dictionaries, 6 formats plus fence)
# riscv_r_fields = {
#     "rd": "11:7",
#     "rs1": "19:15",
#     "rs2": "24:20"
# }

# riscv_i_fields = {
#     "rd": "11:7",
#     "rs1": "19:15",
#     "imm12": "31:20"
# }

# custom_fields_fence = {
#     'fm':   "31:28",
#     'pred': "27:24",
#     'succ': "23:20",
#     'rs1':  "19:15",  
#     'rd':   "11:7"    
# }
# custom_fields_rv32_i = {
#     'rd':   "11:7",
#     'rs1':  "19:15",
#     'shamtw': "24:20"
# }
# custom_field_rv32csr = {
#     'rd':   "11:7",
#     'rs1':  "19:15",
#     'csr': "31:20",
# }
# custom_field_rv32csr_i = {
#     'rd':   "11:7",
#     'csr':  "31:20",
#     'zimm5': "19:15",
# }
# riscv_s_fields = {
#     "imm12hi": "31:25",
#     "rs1": "19:15",
#     "rs2": "24:20",
#     "imm12lo": "11:7"
# }

# riscv_sb_fields = {
#     "bimm12hi": "31:25",
#     "rs1": "19:15",
#     "rs2": "24:20",
#     "bimm12lo": "11:7"
# }

# riscv_u_fields = {
#     "rd": "11:7",
#     "imm20": "31:12"
# }

# riscv_uj_fields = {
#     "rd": "11:7",
#     "jimm20": "31:12"
# }

# unknown_fields = {"UNKNOWN": "still empty"}

#Dictionary with all the formats
# format_dicts = {
    # 'R': riscv_r_fields,
    # 'I': riscv_i_fields,
    # 'S': riscv_s_fields,
    # 'SB': riscv_sb_fields,
    # 'U': riscv_u_fields,
    # 'UJ': riscv_uj_fields,
    # 'FENCE': custom_fields_fence,
    # 'RV32_I': custom_fields_rv32_i,
    # 'RV32CSR': custom_field_rv32csr,
    # 'RV32CSR_I': custom_field_rv32csr_i,
#     'UNKNOWN': unknown_fields
# }

#Dictionary with length of the fields
# field_specs = {
#     "rd":       "[4:0]",
#     "rs1":      "[4:0]",
#     "rs2":      "[4:0]",
#     "imm12":    "[11:0]",
#     "imm12hi":  "[6:0]",
#     "imm12lo":  "[4:0]",
#     "bimm12hi": "[6:0]",
#     "bimm12lo": "[4:0]",
#     "jimm20":   "[19:0]",
#     "imm20":    "[19:0]",
#     "fm":       "[3:0]",
#     "pred":     "[3:0]",
#     "succ":     "[3:0]",
#     "imms":     "[11:0]",
#     "immsb":    "[12:0]",
#     "immuj":    "[31:0]",
#     "pc":       "[31:0]",
#     "reg_mul":  "[63:0]",
#     "shamtw":   "[4:0]",
#     "csr":      "[11:0]",
#     "zimm5":    "[4:0]",
#     "reg_file[31:0]": "[31:0]"
# }

#Opening json to extract parameters to format the template
with open(config_json) as f:            
    config = json.load(f)

#Opening json to extract the opcode's variable fields
with open(instr_dict_json) as f1:         
    instr_dict = json.load(f1)

#Opening json to extract the implementation of each instruction
with open(impl_dict_json) as f2:
    impl_dict = json.load(f2)

#Opening csv to extract the opcode's bit encoding
with open(arg_lut_file) as f3:
    arg_lut = {row[0].strip('" '): (int(row[1]), int(row[2])) for row in csv.reader(f3)}

#Opening json to extract the opcode's priority
with open(opcode_priority) as f4:
    priority_list = json.load(f4)


# def concat_indent(base_indent: str, times: int) -> str:
#     """
#     Concatenates the base indentation string a specified number of times.

#     Args:
#         base_indent (str): The base indentation string.
#         times (int): The number of times to concatenate the base indentation.

#     Returns:
#         str: The concatenated indentation string.
#     """
#     return base_indent * times


#Extracting parameters from config.json to format the template
values = {                                       
    "class_name": config["name"],
    "parent": config["parent"],
    "main_class": config["main_class"],
    "instr_width": config["instr_width"],
    "path_name": config["path_name"],
}


#Extracting the field names and sizes from the field_specs dictionary
field_block = "".join(
    f"{f'bit [{start-end}:0]' if start != end else f'bit'} {field};\n{INDENT_ONE}"
    for field, (start, end) in arg_lut.items()
)

#Using regex to extract the instructions' names and bit encoding from the input file
if os.path.exists(input_file):
    
    with open(input_file, 'r') as file:
        lines = file.readlines()

    extract_opcode = re.compile(r'^\s*localparam\s+\[31:0\]\s+(\w+)\s*=\s*(\S+);')

    for line in lines:
        match = extract_opcode.match(line)
        if match:
            opcode_name = match.group(1)      
            bit_encoding = match.group(2)    
            opcode_dict[opcode_name] = bit_encoding  


#Extracting the variable fields for each instruction from the instr_dict.json
for instruction, data in instr_dict.items():                
    for i, (key, value) in enumerate(data.items()):
        if key == "variable_fields":
            only_variable_fields[instruction] = value    
                   
#Extracting the implementation for each instruction from the impl_dict.json
for instruction, impls in impl_dict.items():
    for i, (key, value) in enumerate(impls.items()):
            implementations_dict[instruction] = value


# Here I use function set to convert the field list in a set, so I can compare the dictionary created before
# for instr, fields in only_variable_fields.items():
#     fset = set(fields)
    # if fset <= set(riscv_r_fields.keys()):
    #     instruction_formats[instr] = 'R'
    # elif fset <= set(riscv_i_fields.keys()) :
    #     instruction_formats[instr] = 'I'
    # if fset <= set(riscv_s_fields.keys()):
    #     instruction_formats[instr] = 'S'
    # elif fset <= set(riscv_sb_fields.keys()):
    #     instruction_formats[instr] = 'SB'
    # elif fset <= set(riscv_u_fields.keys()):
    #     instruction_formats[instr] = 'U'
    # elif fset <= set(riscv_uj_fields.keys()):
    #     instruction_formats[instr] = 'UJ'
    # elif fset <= set(custom_fields_fence.keys()):
    #     instruction_formats[instr] = 'FENCE'
    # elif fset <= set(custom_fields_rv32_i.keys()):
    #     instruction_formats[instr] = 'RV32_I'
    # elif fset <= set(custom_field_rv32csr.keys()):
    #     instruction_formats[instr] = 'RV32CSR'
    # elif fset <= set(custom_field_rv32csr_i.keys()):
    #     instruction_formats[instr] = 'RV32CSR_I'
    # else:
    #     instruction_formats[instr] = 'UNKNOWN'






# #Here I create a dictionary with the bitfield mapping for each variable field
# for instr, fields in only_variable_fields.items():
#     fmt_name = instruction_formats[instr] #this should be only the instruction's format
#     fmt_dict = format_dicts[fmt_name] #This is extracting the variable fields from the dictionary
#     bitfield_mapping[instr] = {
#         field: fmt_dict[field] for field in fields 
#     }




# Filling the casez_dict which will contain all the datas to be put in the case
for i, (key, val) in enumerate(opcode_dict.items()):
    casez_dict[f"condition{i}"] = f"{key}"
    casez_dict[f"assign{i}"]    = f'`uvm_info("{key}", \"Instruction {key} detected successfully\", UVM_LOW)\n'
    for j, (instr, fields) in enumerate(only_variable_fields.items()):
        if(key.lower() == instr.lower()):
            #fmt_name = instruction_formats[instr]
            for var_field, (start, end) in arg_lut.items():
                for single_field in fields:
                    if single_field == var_field:
                        if start == end:
                            casez_dict[f"assign{i}"] += f"{INDENT_THREE}{INDENT_ONE}{var_field} = instr[{start}];\n"
                        else:
                            casez_dict[f"assign{i}"] += f"{INDENT_THREE}{INDENT_ONE}{var_field} = instr[{start}:{end}];\n"
            # if(fmt_name == "S"):
            #     casez_dict[f"assign{i}"] += f"{INDENT_THREE}{INDENT_ONE}imms = {{imm12hi, imm12lo}};\n"
            # if(fmt_name == "SB"):
            #     casez_dict[f"assign{i}"] += f"{INDENT_THREE}{INDENT_ONE}immsb = {{bimm12hi[6], bimm12lo[0], bimm12hi[5:0], bimm12lo[4:1], 1'b0}};\n"
            # if(fmt_name == "UJ"):
            #     casez_dict[f"assign{i}"] += f"{INDENT_THREE}{INDENT_ONE}immuj = {{{{11{{jimm20[19]}}}},jimm20[19], jimm20[7:0], jimm20[8], jimm20[18:9], 1'b0}};\n"
            if instr.lower() in implementations_dict.keys():
                casez_dict[f"assign{i}"] += f"{INDENT_THREE}{INDENT_ONE}{implementations_dict[instr.lower()]}\n"

#Reordering the casez_dict based on the priority list
casez_dict = reorder_casez_dict(casez_dict, priority_list)



#this block manages the cases in which there is the clock or not
if mode == "clock":
    clock_code = """
    task run_phase(uvm_phase phase);

        super.run_phase(phase);

        if (!uvm_config_db#(virtual uvma_clknrst_if)::get(null, "*.env.clknrst_agent", "vif", clknrst_vif)) begin
            `uvm_fatal("NOCLOCK", "Cannot get clknrst_vif from config_db")
        end

        fork
            begin : fetch_decode
                forever begin
                    @(posedge clknrst_vif.clk);

                    if (!clknrst_vif.reset_n) begin
                        pc = 0;
                    end else begin
                        instruction = {mem[pc+3][7:0], mem[pc+2][7:0], mem[pc+1][7:0], mem[pc][7:0]};
                        pc = decode_opcode(instruction, pc);
                        m_analysis_port.write(rvfi_instr_seq_item);
                    end
                end
            end
        join_none
    endtask
"""
    while_code = ""  # Void: I'm not writing anything inside the constructor
    step_code = """
    function uvma_rvfi_instr_seq_item_c#(ILEN,XLEN) step (int i, uvma_rvfi_instr_seq_item_c#(ILEN,XLEN) t);
        //instruction = {mem[pc+3][7:0], mem[pc+2][7:0], mem[pc+1][7:0], mem[pc][7:0]};
        //pc = decode_opcode(instruction, pc);
        //`uvm_info(get_type_name(), "Dummy step function called", UVM_DEBUG)
    endfunction 

    function void write_rvfi_instr(uvma_rvfi_instr_seq_item_c#(ILEN,XLEN) t);
        //uvma_rvfi_instr_seq_item_c#(ILEN,XLEN) t_reference_model = step(1, t);
        //m_analysis_port.write(t);
        //`uvm_info(get_type_name(), "Dummy write_rvfi_instr function called", UVM_DEBUG)
    endfunction : write_rvfi_instr
"""
elif mode == "while":
    clock_code = ""    # Void: I'm not writing anything inside the run_phase
    while_code = """
        while (pc != 32'h80000288) begin
            instruction = {mem[pc+3][7:0], mem[pc+2][7:0], mem[pc+1][7:0], mem[pc][7:0]};
            pc = decode_opcode(instruction, pc);
            m_analysis_port.write(rvfi_instr_seq_item);
        end
"""
    step_code = """
    function uvma_rvfi_instr_seq_item_c#(ILEN,XLEN) step (int i, uvma_rvfi_instr_seq_item_c#(ILEN,XLEN) t);
        //instruction = {mem[pc+3][7:0], mem[pc+2][7:0], mem[pc+1][7:0], mem[pc][7:0]};
        //pc = decode_opcode(instruction, pc);
        //`uvm_info(get_type_name(), "Dummy step function called", UVM_DEBUG)
    endfunction 

    function void write_rvfi_instr(uvma_rvfi_instr_seq_item_c#(ILEN,XLEN) t);
        //uvma_rvfi_instr_seq_item_c#(ILEN,XLEN) t_reference_model = step(1, t);
        //m_analysis_port.write(t);
        //`uvm_info(get_type_name(), "Dummy write_rvfi_instr function called", UVM_DEBUG)
    endfunction : write_rvfi_instr
"""
else:
    clock_code = ""    # Void: I'm not writing anything inside the run_phase
    while_code = ""    # Void: I'm not writing anything inside the run_phase
    step_code = """
    function uvma_rvfi_instr_seq_item_c#(ILEN,XLEN) step (int i, uvma_rvfi_instr_seq_item_c#(ILEN,XLEN) t);
        uvma_rvfi_instr_seq_item_c#(ILEN,XLEN) t_reference_model_prov;
        instruction = {mem[pc+3][7:0], mem[pc+2][7:0], mem[pc+1][7:0], mem[pc][7:0]};
        t_reference_model_prov = decode_opcode(instruction);
        `uvm_info(get_type_name(), "Dummy step function called", UVM_DEBUG)
        return t_reference_model_prov;
    endfunction 

    function void write_rvfi_instr(uvma_rvfi_instr_seq_item_c#(ILEN,XLEN) t);
        uvma_rvfi_instr_seq_item_c#(ILEN,XLEN) t_reference_model = step(1, t);
        m_analysis_port.write(t_reference_model);
        `uvm_info(get_type_name(), "Dummy write_rvfi_instr function called", UVM_DEBUG)
    endfunction : write_rvfi_instr
"""

#This block is used to fill the rvfi_instr_seq_item with the values extracted from the instruction
rvfi_block = f"""
{INDENT_TWO}rvfi_instr_seq_item.order     = order++;
{INDENT_TWO}rvfi_instr_seq_item.insn      = instr;
{INDENT_TWO}rvfi_instr_seq_item.rs1_addr  = rs1;
{INDENT_TWO}rvfi_instr_seq_item.rs1_rdata = reg_rs1_prev;
{INDENT_TWO}rvfi_instr_seq_item.rs2_addr  = rs2;
{INDENT_TWO}rvfi_instr_seq_item.rs2_rdata = reg_rs2_prev;
{INDENT_TWO}rvfi_instr_seq_item.rd1_addr  = rd;
{INDENT_TWO}rvfi_instr_seq_item.rd1_wdata = reg_file[rd];
{INDENT_TWO}rvfi_instr_seq_item.pc_rdata  = pc_before;
{INDENT_TWO}rvfi_instr_seq_item.pc_wdata  = pc;
"""


#Class template to be formatted
template_content = """
`ifndef __{class_name}_SV__
`define __{class_name}_SV__

import riscv_instr::*;
import uvma_rvfi_pkg::*;

class {class_name} extends {main_class};

    string {path_name} = "";
    int mem[int];
    int incr;
    int order = 0;

    virtual uvma_clknrst_if clknrst_vif;
    uvma_rvfi_mode mode = 3;

    {fields_variables}
    //Added by hand (not present in arg_lut.csv)
    bit [31:0] instruction;
    bit [31:0] reg_file[31:0];
    bit [31:0] csr_reg_file[4095:0];
    bit [11:0] imms;
    bit [12:0] immsb; 
    bit [31:0] immuj; 
    bit [31:0] pc; 
    bit [63:0] reg_mul;
    bit [31:0] imm12_ext;
    bit [31:0] imms_ext;
    bit [31:0] immsb_ext;
    bit [31:0] pc_before;
    bit [31:0] c_imm_ext;
    bit [31:0] addr;
    bit [31:0] reg_rs1_prev;
    bit [31:0] reg_rs2_prev;
    bit [31:0] rs2_masked;
    bit [31:0] imm6_ext;
    bit [31:0] reg_result;
    bit iteration_mx = 0;
    bit [15:0] reg_mac_mul_prov;


    uvma_rvfi_instr_seq_item_c#(32, 32) rvfi_instr_seq_item;
    `uvm_component_utils_begin({class_name})
    `uvm_component_utils_end

    function new(string name="{class_name}", uvm_component parent={parent});

        super.new(name, parent);

        $display("[%0t]Creating {class_name} instance: %s", $time, name);

	    if ($value$plusargs("firmware=%s", {path_name})) begin
            $display("Firmware file: %s", {path_name});
        end else begin
            $fatal("No +firmware argument provided!");
    	end

        $readmemh({path_name}, mem);

        csr_reg_file[12'hF11] = 32'h00000602; // mvendorid
        csr_reg_file[12'h301] = 32'h40101104; // misa (RV32IMCU)
        csr_reg_file[12'hF12] = 32'h00000023; // marchid
        csr_reg_file[12'hF13] = 32'h00000000; // mimpid
        csr_reg_file[12'h300] = 32'h00001800; // mstatus

    endfunction : new

    function void build_phase(uvm_phase phase);
        st_core_cntrl_cfg st;

        super.build_phase(phase);

        st = cfg.to_struct();

        if (st.boot_addr_valid) begin
            pc = st.boot_addr;
            `uvm_info("Boot_addr: %h", st.boot_addr, UVM_LOW)
        end else begin
            `uvm_fatal("Boot_addr not valid, using default value", UVM_LOW)
        end
        {constructor_code}
    endfunction : build_phase

    {step_code}
    {run_phase_code}

    function uvma_rvfi_instr_seq_item_c#(ILEN,XLEN) decode_opcode(bit[{instr_width}:0] instr);

        rvfi_instr_seq_item = uvma_rvfi_instr_seq_item_c#(32,32)::type_id::create("rvfi_instr_seq_item", this);

        rvfi_instr_seq_item.mode = mode;

        incr = 4;

        pc_before = pc;

        rs1 = 5'b0;
        rs2 = 5'b0;
        rd  = 5'b0;

    {casez_string}

        reg_file[0] = 32'b0;

        pc += incr;

        {rvfi_block}

        return rvfi_instr_seq_item;

    endfunction : decode_opcode


endclass : {class_name}

`endif // __{class_name}_SV__
"""





#Formatting the template with the extracted parameters
casez_fmt = get_if_else_statement_fmt(length=len(opcode_dict)-1, case_format=True, always_comb=False)
    
casez_string = casez_fmt.format(
    indent="        ",
    val="instr",
    default_assign= f"begin\n\n{INDENT_THREE}{INDENT_ONE}`uvm_error(\"UNKNOWN\", \"Unknown instruction detected\")\n{INDENT_THREE}{INDENT_ONE}incr = 4;\n\n{INDENT_THREE}end\n",
    **casez_dict,
)

file_content = template_content.format(casez_string=casez_string,**values, fields_variables=field_block, constructor_code=while_code, run_phase_code=clock_code, step_code=step_code, rvfi_block=rvfi_block)



#Writing the formatted content to the sysverilog class
directory = "../lib/uvm_components/uvmc_rvfi_reference_model"

output_file = os.path.join(directory, config["name"] + ".sv")

with open(output_file, 'w') as file:
    file.write(file_content)

print(f"File content written to {output_file}")