import os
import re
import json


#Valerio's if/else/case format function
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


only_variable_fields = dict()  #Dictionary which will contain key = instruction's name and val = variable fields
opcode_dict = dict()           #Dictionary which will contain key = instruction's name and val = instruction's bit encoding
instruction_formats = dict()   #Dictionary which will contain key = instruction's name and val = instruction's type (R, I, S, ecc.)
bitfield_mapping = dict()      #Dictionary which will contain key = instruction's name and val = dictionary with the bitfield mapping for each variable fields
casez_dict = dict()            #Dictionary which will contain key = instruction's name and val = all the stuff to be put in the each case statement
implementations_dict = dict()  #Dictionary which will contain key = instruction's name and val = the implementation of the instruction


# Define field dictionaries for different instruction types (7 dictionaries, 6 formats but FENCE sucks)
riscv_r_fields = {
    "rd": "11:7",
    "rs1": "19:15",
    "rs2": "24:20"
}

riscv_i_fields = {
    "rd": "11:7",
    "rs1": "19:15",
    "imm12": "31:20"
}

custom_fields_fence = {
    'fm':   "31:28",
    'pred': "27:24",
    'succ': "23:20",
    'rs1':  "19:15",  
    'rd':   "11:7"    
}
riscv_s_fields = {
    "imm12hi": "31:25",
    "rs1": "19:15",
    "rs2": "24:20",
    "imm12lo": "11:7"
}

riscv_sb_fields = {
    "bimm12hi": "31:25",
    "rs1": "19:15",
    "rs2": "24:20",
    "bimm12lo": "11:7"
}

riscv_u_fields = {
    "rd": "11:7",
    "imm20": "31:12"
}

riscv_uj_fields = {
    "rd": "11:7",
    "jimm20": "31:12"
}


#Dictionary with all the formats
format_dicts = {
    'R': riscv_r_fields,
    'I': riscv_i_fields,
    'S': riscv_s_fields,
    'SB': riscv_sb_fields,
    'U': riscv_u_fields,
    'UJ': riscv_uj_fields,
    'FENCE': custom_fields_fence
}

#Dictionary with length of the fields
field_specs = {
    "rd":       "[4:0]",
    "rs1":      "[4:0]",
    "rs2":      "[4:0]",
    "imm12":    "[11:0]",
    "imm12hi":  "[6:0]",
    "imm12lo":  "[4:0]",
    "bimm12hi": "[6:0]",
    "bimm12lo": "[4:0]",
    "jimm20":   "[19:0]",
    "imm20":    "[19:0]",
    "fm":       "[3:0]",
    "pred":     "[3:0]",
    "succ":     "[3:0]",
    "imms":     "[11:0]",
    "immsb":    "[12:0]",
    "immuj":    "[20:0]",
    "pc":       "[31:0]",
    "reg_mul":  "[63:0]",
    "reg_file[31:0]": "[31:0]"
}

#Opening json to extract parameters to format the template
with open(config_json) as f:            
    config = json.load(f)

#Opening json to extract the opcode's variable fields
with open(instr_dict_json) as f1:         
    instr_dict = json.load(f1)

#Opening json to extract the implementation of each instruction
with open(impl_dict_json) as f2:
    impl_dict = json.load(f2)

#Global variables to make an indentation when needed
INDENT_ONE = "    "                     
INDENT_TWO = "        "
INDENT_THREE = "            "

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
    "nome_path": config["nome_path"]
}


#Extracting the field names and sizes from the field_specs dictionary
field_block = "".join(f"bit {size} {name};\n{INDENT_ONE}" for name, size in field_specs.items())

#Using regex to extract the instructions' names and bit encoding from the input file
if os.path.exists(input_file):
    
    with open(input_file, 'r') as file:
        lines = file.readlines()

    opcode_estratto = re.compile(r'^\s*localparam\s+\[31:0\]\s+(\w+)\s*=\s*(\S+);')

    for line in lines:
        match = opcode_estratto.match(line)
        if match:
            nome_opcode = match.group(1)      
            valore_opcode = match.group(2)    
            opcode_dict[nome_opcode] = valore_opcode  







#Extracting only variable fields for each instruction from the instr_dict.json
for instruction, dati in instr_dict.items():                
    for i, (chiave, valore) in enumerate(dati.items()):
        if chiave == "variable_fields":
            only_variable_fields[instruction] = valore

#Extracting the implementation of each instruction from the impl_dict.json
for instruction, impls in impl_dict.items():
    for i, (chiave, valore) in enumerate(impls.items()):
            implementations_dict[instruction] = valore


# Here I use function set to convert the field list in a set, so I can compare the dictionary created before
for instr, fields in only_variable_fields.items():
    fset = set(fields)
    if fset <= set(riscv_r_fields.keys()):
        instruction_formats[instr] = 'R'
    elif fset <= set(riscv_i_fields.keys()) :
        instruction_formats[instr] = 'I'
    elif fset <= set(riscv_s_fields.keys()):
        instruction_formats[instr] = 'S'
    elif fset <= set(riscv_sb_fields.keys()):
        instruction_formats[instr] = 'SB'
    elif fset <= set(riscv_u_fields.keys()):
        instruction_formats[instr] = 'U'
    elif fset <= set(riscv_uj_fields.keys()):
        instruction_formats[instr] = 'UJ'
    elif fset <= set(custom_fields_fence.keys()):
        instruction_formats[instr] = 'FENCE'
    else:
        instruction_formats[instr] = 'UNKNOWN'






#Here I create a dictionary with the bitfield mapping for each variable field
for instr, fields in only_variable_fields.items():
    fmt_name = instruction_formats[instr] #this should be only the instruction's format
    fmt_dict = format_dicts[fmt_name] #This is extracting the variable fields from the dictionary
    bitfield_mapping[instr] = {
        field: fmt_dict[field] for field in fields 
    }




#Here I fill the casez_dict which will contain all the stuff to be put in the case
for i, (key, val) in enumerate(opcode_dict.items()):
    casez_dict[f"condition{i}"] = f"{key}"
    casez_dict[f"assign{i}"]    = f'`uvm_info("{key}", \"Instruction {key} detected successfully\", UVM_LOW)\n'
    for j, (instr, fields) in enumerate(only_variable_fields.items()):
        if(key.lower() == instr):
            fmt_name = instruction_formats[instr]
            for instruct, start_end in bitfield_mapping[instr].items():
                casez_dict[f"assign{i}"] += f"{INDENT_THREE}{INDENT_ONE}{instruct} = instr[{start_end}];\n"
            if(fmt_name == "S"):
                casez_dict[f"assign{i}"] += f"{INDENT_THREE}{INDENT_ONE}imms = {{imm12hi, imm12lo}};\n"
            if(fmt_name == "SB"):
                casez_dict[f"assign{i}"] += f"{INDENT_THREE}{INDENT_ONE}immsb = {{bimm12hi[6], bimm12lo[0], bimm12hi[5:0], bimm12lo[4:1], 1'b0}};\n"
            if(fmt_name == "UJ"):
                casez_dict[f"assign{i}"] += f"{INDENT_THREE}{INDENT_ONE}immuj = {{jimm20[19], jimm20[7:0], jimm20[8], jimm20[18:9], 1'b0}};\n"
            if(instr) in implementations_dict.keys():
                casez_dict[f"assign{i}"] += f"{INDENT_THREE}{INDENT_ONE}{implementations_dict[instr]};\n"


#Class template to be formatted
template_content = """
`ifndef __{class_name}_SV__
`define __{class_name}_SV__

import riscv_instr::*;

class {class_name} extends {main_class};

    string {nome_path} = "";
    int mem[int];

    {fields_variables}

    `uvm_component_utils_begin({class_name})
    `uvm_component_utils_end

    function new(string name="{class_name}", {main_class} parent={parent});

        super.new(name, parent);

        $display("[%0t]Creating {class_name} instance: %s", $time, name);

	    if ($value$plusargs("firmware=%s", {nome_path})) begin
            $display("Firmware file: %s", {nome_path});
    	end else begin
            $fatal("No +firmware argument provided!");
    	end

        $readmemh({nome_path}, mem);

        foreach (mem[i]) begin
            decode_opcode(mem[i]);
        end

    endfunction : new



    function void decode_opcode(bit[{instr_width}:0] instr);

    {casez_string}

    endfunction : decode_opcode


endclass : {class_name}

`endif // __{class_name}_SV__
"""





#Formatting the template with the extracted parameters
casez_fmt = get_if_else_statement_fmt(length=len(opcode_dict)-1, case_format=True, always_comb=False)
    
casez_string = casez_fmt.format(
    indent="        ",
    val="instr",
    default_assign= "`uvm_error(\"UNKNOWN\", \"Unknown instruction detected\")",
    **casez_dict,
)

file_content = template_content.format(casez_string=casez_string,**values, fields_variables=field_block,)



#Writing the formatted content to the sysverilog class
directory = "/home/pab/workspace/core-v-verif/lib/uvm_components/uvmc_rvfi_reference_model"

output_file = os.path.join(directory, config["name"] + ".sv")

with open(output_file, 'w') as file:
    file.write(file_content)

print(f"File content written to {output_file}")