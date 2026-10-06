// ============================================================================
// Module: triton_systolic_tile
// Description: 4x4 Systolic Matrix Engine for Triton-X100 Tile Execution Unit.
//              Computes C[4x4] += A[4xK] * B[Kx4] in hardware.
// ============================================================================

`timescale 1ns / 1ps

module triton_systolic_tile #(
    parameter DATA_WIDTH = 8,
    parameter ACC_WIDTH  = 32,
    parameter TILE_SIZE  = 4
)(
    input  wire                               clk,
    input  wire                               rst_n,
    input  wire                               enable,
    input  wire                               clear_acc,
    // Matrix A inputs (Row feed)
    input  wire [TILE_SIZE*DATA_WIDTH-1:0]    a_row_in,
    // Matrix B inputs (Col feed)
    input  wire [TILE_SIZE*DATA_WIDTH-1:0]    b_col_in,
    // Flattened Matrix C outputs (4x4 x 32-bit = 512 bits)
    output wire [TILE_SIZE*TILE_SIZE*ACC_WIDTH-1:0] c_tile_out
);

    // Internal systolic interconnect wires
    wire [DATA_WIDTH-1:0] a_wires [0:TILE_SIZE-1][0:TILE_SIZE];
    wire [DATA_WIDTH-1:0] b_wires [0:TILE_SIZE][0:TILE_SIZE-1];

    // Connect boundary inputs
    genvar r, c;
    generate
        for (r = 0; r < TILE_SIZE; r = r + 1) begin : gen_a_in
            assign a_wires[r][0] = a_row_in[(r+1)*DATA_WIDTH-1 : r*DATA_WIDTH];
        end
        for (c = 0; c < TILE_SIZE; c = c + 1) begin : gen_b_in
            assign b_wires[0][c] = b_col_in[(c+1)*DATA_WIDTH-1 : c*DATA_WIDTH];
        end
    endgenerate

    // 2D Array of Processing Elements
    generate
        for (r = 0; r < TILE_SIZE; r = r + 1) begin : gen_row
            for (c = 0; c < TILE_SIZE; c = c + 1) begin : gen_col
                wire [ACC_WIDTH-1:0] pe_acc;
                triton_mac_pe #(
                    .DATA_WIDTH(DATA_WIDTH),
                    .ACC_WIDTH(ACC_WIDTH)
                ) pe_inst (
                    .clk(clk),
                    .rst_n(rst_n),
                    .enable(enable),
                    .clear_acc(clear_acc),
                    .a_in(a_wires[r][c]),
                    .b_in(b_wires[r][c]),
                    .a_out(a_wires[r][c+1]),
                    .b_out(b_wires[r+1][c]),
                    .acc_out(pe_acc)
                );
                // Assign to flattened output
                assign c_tile_out[((r*TILE_SIZE + c)+1)*ACC_WIDTH-1 : (r*TILE_SIZE + c)*ACC_WIDTH] = pe_acc;
            end
        end
    endgenerate

endmodule
