using Mocking
using Test

function random_number()
    return @mock rand()
end

@testset begin
    
    Mocking.activate()

    patch = @patch rand() = 0.2
    apply(patch) do
        println(random_number())
        @test random_number() == 0.2
    end
end