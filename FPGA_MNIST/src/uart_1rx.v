module uart_rx(
    input sys_clk,
    input sys_rst_n,
    input rx_data,

    output rx_done,
    output reg [7:0] send_data
    );


reg [7:0] baud_cnt;
reg [3:0] bit_cnt;
reg rx_data_d0;
reg rx_data_d1;
wire start_judge;
reg rx_en;

// 打两拍
always @ (posedge sys_clk or negedge sys_rst_n) begin
    if(!sys_rst_n) begin
        rx_data_d0 <= 1'b0;
        rx_data_d1 <= 1'b0;
        end
    else begin
        rx_data_d0 <= rx_data;
        rx_data_d1 <= rx_data_d0;
        end
end

// 确定接受态1
assign start_judge = rx_data_d1 & (~rx_data_d0) & ~rx_en;
//assign rx_en = start_judge == 1'b1 ? 1'b1 : rx_en;
// 确定接受态2:
always @ (posedge sys_clk or negedge sys_rst_n) begin
    if(!sys_rst_n)
        rx_en <= 1'b0;
    else if(start_judge)
        rx_en <= 1'b1;
    else if(bit_cnt == 9 && baud_cnt == 8'd109)
        rx_en <= 1'b0;
end

assign rx_done = (bit_cnt == 9 && baud_cnt == 8'd109);

// 
always @ (posedge sys_clk or negedge sys_rst_n) begin
    if(!sys_rst_n) begin
        send_data <= 8'b0;
        baud_cnt <= 8'b0;
        bit_cnt <= 4'b0;
        end
    else if(rx_en) begin
        baud_cnt <= baud_cnt + 8'd1;
        if(baud_cnt >= 8'd217 && bit_cnt < 4'd9) begin
            baud_cnt <= 8'd0;
            bit_cnt <= ((bit_cnt>=9) ? 1'b0 : (bit_cnt + 4'd1));
            end
        else if(bit_cnt == 4'd9 && baud_cnt == 8'd109) begin
            baud_cnt <= 8'd0;
            bit_cnt <= ((bit_cnt>=9) ? 1'b0 : (bit_cnt + 4'd1));
            end
        else if(baud_cnt == 8'd109) begin
            case(bit_cnt)
                4'd1 : send_data[0] <= rx_data_d1;
                4'd2 : send_data[1] <= rx_data_d1;
                4'd3 : send_data[2] <= rx_data_d1;
                4'd4 : send_data[3] <= rx_data_d1;
                4'd5 : send_data[4] <= rx_data_d1;
                4'd6 : send_data[5] <= rx_data_d1;
                4'd7 : send_data[6] <= rx_data_d1;
                4'd8 : send_data[7] <= rx_data_d1;
                default : ;
            endcase
            end
        else
            send_data <= send_data;
    end
end

endmodule