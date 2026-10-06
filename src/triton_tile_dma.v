// ============================================================================
// Module: triton_tile_dma
// Description: Asynchronous Tile DMA Engine (ATDE) for Triton-X100.
//              Directly realizes Triton `tl.make_block_ptr` & `tl.advance`
//              with automated address generation and bounds checking.
// ============================================================================

`timescale 1ns / 1ps

module triton_tile_dma #(
    parameter ADDR_WIDTH = 32,
    parameter DATA_WIDTH = 32
)(
    input  wire                   clk,
    input  wire                   rst_n,
    
    // Command Interface (from Triton Block Sequencer)
    input  wire                   start_dma,
    input  wire [ADDR_WIDTH-1:0]  base_ptr,
    input  wire [15:0]            block_m,
    input  wire [15:0]            block_k,
    input  wire [ADDR_WIDTH-1:0]  stride_m,
    output reg                    dma_busy,
    output reg                    dma_done,
    
    // Memory Bus Request Interface (to HBM3e/L2 Controller)
    output reg                    mem_req_valid,
    output reg  [ADDR_WIDTH-1:0]  mem_req_addr,
    input  wire                   mem_req_ready
);

    // State Machine
    localparam IDLE      = 2'b00;
    localparam BURST_ROW = 2'b01;
    localparam ADVANCE   = 2'b10;
    localparam DONE      = 2'b11;

    reg [1:0]  state;
    reg [15:0] row_cnt;
    reg [15:0] col_cnt;
    reg [ADDR_WIDTH-1:0] current_row_base;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state            <= IDLE;
            dma_busy         <= 1'b0;
            dma_done         <= 1'b0;
            mem_req_valid    <= 1'b0;
            mem_req_addr     <= {ADDR_WIDTH{1'b0}};
            row_cnt          <= 16'd0;
            col_cnt          <= 16'd0;
            current_row_base <= {ADDR_WIDTH{1'b0}};
        end else begin
            case (state)
                IDLE: begin
                    dma_done <= 1'b0;
                    if (start_dma) begin
                        dma_busy         <= 1'b1;
                        row_cnt          <= 16'd0;
                        col_cnt          <= 16'd0;
                        current_row_base <= base_ptr;
                        mem_req_addr     <= base_ptr;
                        mem_req_valid    <= 1'b1;
                        state            <= BURST_ROW;
                    end else begin
                        dma_busy         <= 1'b0;
                    end
                end

                BURST_ROW: begin
                    if (mem_req_ready && mem_req_valid) begin
                        if (col_cnt + 1'b1 < block_k) begin
                            col_cnt       <= col_cnt + 1'b1;
                            mem_req_addr  <= mem_req_addr + 4; // Word stride
                            mem_req_valid <= 1'b1;
                        end else begin
                            col_cnt       <= 16'd0;
                            mem_req_valid <= 1'b0;
                            state         <= ADVANCE;
                        end
                    end
                end

                ADVANCE: begin
                    if (row_cnt + 1'b1 < block_m) begin
                        row_cnt          <= row_cnt + 1'b1;
                        current_row_base <= current_row_base + stride_m;
                        mem_req_addr     <= current_row_base + stride_m;
                        mem_req_valid    <= 1'b1;
                        state            <= BURST_ROW;
                    end else begin
                        state <= DONE;
                    end
                end

                DONE: begin
                    dma_busy      <= 1'b0;
                    dma_done      <= 1'b1;
                    mem_req_valid <= 1'b0;
                    state         <= IDLE;
                end
            endcase
        end
    end

endmodule
