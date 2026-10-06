// ============================================================================
// Module: triton_teu_top
// Description: Top-level Tile Execution Unit (TEU) for Triton-X100.
//              Combines Systolic Matrix Engine and Asynchronous Tile DMA.
// ============================================================================

`timescale 1ns / 1ps

module triton_teu_top #(
    parameter DATA_WIDTH = 8,
    parameter ACC_WIDTH  = 32,
    parameter TILE_SIZE  = 4,
    parameter ADDR_WIDTH = 32
)(
    input  wire                               clk,
    input  wire                               rst_n,
    
    // Control
    input  wire                               compute_en,
    input  wire                               clear_acc,
    
    // Matrix inputs (systolic feeds)
    input  wire [TILE_SIZE*DATA_WIDTH-1:0]    a_row_in,
    input  wire [TILE_SIZE*DATA_WIDTH-1:0]    b_col_in,
    
    // Output matrix tile
    output wire [TILE_SIZE*TILE_SIZE*ACC_WIDTH-1:0] c_tile_out,
    
    // DMA Subsystem
    input  wire                               dma_start,
    input  wire [ADDR_WIDTH-1:0]              dma_base_ptr,
    input  wire [15:0]                        dma_block_m,
    input  wire [15:0]                        dma_block_k,
    input  wire [ADDR_WIDTH-1:0]              dma_stride_m,
    output wire                               dma_busy,
    output wire                               dma_done,
    
    // Memory Bus
    output wire                               mem_req_valid,
    output wire [ADDR_WIDTH-1:0]              mem_req_addr,
    input  wire                               mem_req_ready
);

    // Instantiate Systolic Matrix Engine
    triton_systolic_tile #(
        .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH(ACC_WIDTH),
        .TILE_SIZE(TILE_SIZE)
    ) u_systolic (
        .clk(clk),
        .rst_n(rst_n),
        .enable(compute_en),
        .clear_acc(clear_acc),
        .a_row_in(a_row_in),
        .b_col_in(b_col_in),
        .c_tile_out(c_tile_out)
    );

    // Instantiate Tile DMA Engine
    triton_tile_dma #(
        .ADDR_WIDTH(ADDR_WIDTH),
        .DATA_WIDTH(DATA_WIDTH)
    ) u_dma (
        .clk(clk),
        .rst_n(rst_n),
        .start_dma(dma_start),
        .base_ptr(dma_base_ptr),
        .block_m(dma_block_m),
        .block_k(dma_block_k),
        .stride_m(dma_stride_m),
        .dma_busy(dma_busy),
        .dma_done(dma_done),
        .mem_req_valid(mem_req_valid),
        .mem_req_addr(mem_req_addr),
        .mem_req_ready(mem_req_ready)
    );

endmodule
